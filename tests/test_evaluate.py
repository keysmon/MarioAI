from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import marioai.evaluate as evaluation
from marioai.evaluate import evaluate_checkpoint, evaluate_rollout
from marioai.levels import ALL_LEVELS
from marioai.results import (
    EvaluationReport,
    RolloutResult,
    checkpoint_score,
    is_better_checkpoint,
    sha256_file,
    summarize_stage,
)


def _rollout(
    level: str,
    seed: int,
    *,
    cleared: bool = False,
    max_x: int = 100,
) -> RolloutResult:
    return RolloutResult(
        level=level,
        seed=seed,
        cleared=cleared,
        terminal_cause="flag" if cleared else "death",
        max_x=max_x,
        reward=1.0,
        steps=10,
        wall_seconds=0.1,
    )


def _report(
    *,
    passed_stages: int = 32,
    clears: int = 32,
    mean_max_x: float = 100.0,
) -> EvaluationReport:
    stages = {}
    for index, level in enumerate(ALL_LEVELS):
        passed = index < passed_stages
        stage_clears = clears // passed_stages if passed and passed_stages else 0
        if passed and index < clears % passed_stages:
            stage_clears += 1
        stages[level] = {
            "passed": passed,
            "clears": stage_clears,
            "episodes": 15,
            "clear_rate": stage_clears / 15,
            "mean_max_x": mean_max_x,
            "mean_reward": 1.0,
        }
    return EvaluationReport(
        checkpoint_sha256="a" * 64,
        deterministic=False,
        requested_episodes=15,
        stages=stages,
        rollouts=[],
    )


def test_stage_passes_with_one_of_15_clears():
    rollouts = [
        _rollout("1-1", index, cleared=index == 7)
        for index in range(15)
    ]

    stage = summarize_stage("1-1", rollouts)

    assert stage["passed"] is True
    assert stage["clears"] == 1
    assert stage["episodes"] == 15


def test_project_pass_requires_same_checkpoint_and_all_32_stages():
    report = _report()

    assert report.passed is True
    assert not replace(
        report,
        stages={key: value for key, value in report.stages.items() if key != "8-4"},
    ).passed


def test_acceptance_rejects_deterministic_or_non_15_report():
    report = _report()

    assert not replace(report, deterministic=True).passed
    assert not replace(report, requested_episodes=14).passed


def test_acceptance_rejects_stage_without_15_episodes_or_a_clear():
    report = _report()
    wrong_count = dict(report.stages)
    wrong_count["8-4"] = {**wrong_count["8-4"], "episodes": 14}
    no_clear = dict(report.stages)
    no_clear["8-4"] = {**no_clear["8-4"], "passed": False, "clears": 0}

    assert not replace(report, stages=wrong_count).passed
    assert not replace(report, stages=no_clear).passed


def test_checkpoint_selection_prefers_coverage_then_clears_then_progress():
    report_with_8_stages = _report(passed_stages=8, clears=8)
    report_with_7_stages = _report(passed_stages=7, clears=70, mean_max_x=999.0)
    report_with_8_stages_12_clears = _report(passed_stages=8, clears=12)
    report_with_8_stages_9_clears = _report(
        passed_stages=8, clears=9, mean_max_x=999.0
    )
    report_with_more_progress = _report(
        passed_stages=8, clears=12, mean_max_x=101.0
    )
    report_with_less_progress = _report(
        passed_stages=8, clears=12, mean_max_x=100.0
    )

    assert checkpoint_score(report_with_8_stages) == (8, 8, 3200.0)
    assert is_better_checkpoint(report_with_8_stages, report_with_7_stages)
    assert is_better_checkpoint(
        report_with_8_stages_12_clears,
        report_with_8_stages_9_clears,
    )
    assert is_better_checkpoint(
        report_with_more_progress,
        report_with_less_progress,
    )
    assert is_better_checkpoint(report_with_7_stages, None)


def test_checkpoint_progress_falls_back_to_rollout_evidence():
    report = _report(passed_stages=8, clears=12)
    minimal_stages = {
        level: {
            "passed": stage["passed"],
            "clears": stage["clears"],
            "episodes": stage["episodes"],
        }
        for level, stage in report.stages.items()
    }
    less_progress = replace(
        report,
        stages=minimal_stages,
        rollouts=[_rollout("1-1", 1, max_x=100)],
    )
    more_progress = replace(
        report,
        stages=minimal_stages,
        rollouts=[_rollout("1-1", 1, max_x=101)],
    )

    assert is_better_checkpoint(more_progress, less_progress)


def test_report_json_round_trip_preserves_stable_schema(tmp_path):
    report = EvaluationReport(
        checkpoint_sha256="b" * 64,
        deterministic=False,
        requested_episodes=15,
        stages={"1-1": summarize_stage("1-1", [_rollout("1-1", 42000)])},
        rollouts=[_rollout("1-1", 42000)],
    )
    path = tmp_path / "reports" / "evaluation.json"

    report.write(path)
    payload = json.loads(path.read_text())

    assert payload == {
        "checkpoint_sha256": "b" * 64,
        "deterministic": False,
        "requested_episodes": 15,
        "stages": {
            "1-1": {
                "passed": False,
                "clears": 0,
                "episodes": 1,
                "clear_rate": 0.0,
                "mean_max_x": 100.0,
                "mean_reward": 1.0,
            }
        },
        "rollouts": [
            {
                "level": "1-1",
                "seed": 42000,
                "cleared": False,
                "terminal_cause": "death",
                "max_x": 100,
                "reward": 1.0,
                "steps": 10,
                "wall_seconds": 0.1,
            }
        ],
    }
    assert EvaluationReport.read(path) == report


def test_report_write_replaces_destination_atomically(tmp_path, monkeypatch):
    report = _report()
    path = tmp_path / "evaluation.json"
    replacements = []
    original_replace = type(path).replace

    def recording_replace(source, destination):
        replacements.append((source, destination))
        return original_replace(source, destination)

    monkeypatch.setattr(type(path), "replace", recording_replace)

    report.write(path)

    assert len(replacements) == 1
    temporary, destination = replacements[0]
    assert temporary.parent == path.parent
    assert temporary != path
    assert destination == path
    assert not temporary.exists()


def test_sha256_file_hashes_exact_checkpoint_bytes(tmp_path):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"immutable checkpoint evidence")

    assert sha256_file(checkpoint) == hashlib.sha256(
        b"immutable checkpoint evidence"
    ).hexdigest()


class _SeededPolicy:
    def __init__(self):
        self.sample_seed = None

    def set_random_seed(self, seed):
        self.sample_seed = seed

    def predict(self, observation, *, deterministic):
        assert deterministic is False
        return np.array(
            [(int(observation[0]) + self.sample_seed * 7) % 97]
        ), None


class _OneStepVectorEnv:
    def __init__(self):
        self.reset_seed = None
        self.closed = False

    def seed(self, seed):
        self.reset_seed = seed

    def reset(self):
        return np.array([self.reset_seed * 3])

    def step(self, action):
        x_pos = int(action[0])
        return (
            np.array([0]),
            np.array([2.5]),
            np.array([True]),
            [{"flag_get": False, "time": 200, "x_pos": x_pos}],
        )

    def close(self):
        self.closed = True


def test_rollout_seed_controls_environment_reset_and_policy_sampling(monkeypatch):
    environments = []

    def build_environment(**_kwargs):
        environment = _OneStepVectorEnv()
        environments.append(environment)
        return environment

    monkeypatch.setattr(evaluation, "_make_evaluation_env", build_environment)
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: 10.0)

    first = evaluate_rollout(
        _SeededPolicy(), "1-1", seed=123, deterministic=False
    )
    repeated = evaluate_rollout(
        _SeededPolicy(), "1-1", seed=123, deterministic=False
    )
    different = evaluate_rollout(
        _SeededPolicy(), "1-1", seed=124, deterministic=False
    )

    assert first == repeated
    assert first.max_x == (123 * 3 + 123 * 7) % 97
    assert different.max_x == (124 * 3 + 124 * 7) % 97
    assert different.max_x != first.max_x
    assert all(environment.closed for environment in environments)


@pytest.mark.parametrize(
    ("infos", "max_steps", "expected"),
    [
        ([{"flag_get": True, "time": 0, "x_pos": 1}], 1, "flag"),
        ([{"flag_get": False, "time": 0, "x_pos": 1}], 1, "timeout"),
        ([{"flag_get": False, "time": 0, "x_pos": 1}], 2, "time"),
        ([{"flag_get": False, "time": 200, "x_pos": 1}], 2, "death"),
    ],
)
def test_rollout_terminal_cause_uses_required_priority(
    monkeypatch, infos, max_steps, expected
):
    class TerminalEnv(_OneStepVectorEnv):
        def step(self, action):
            return (
                np.array([0]),
                np.array([0.0]),
                np.array([True]),
                infos,
            )

    monkeypatch.setattr(
        evaluation, "_make_evaluation_env", lambda **_kwargs: TerminalEnv()
    )

    result = evaluate_rollout(
        _SeededPolicy(),
        "1-1",
        seed=5,
        deterministic=False,
        max_steps=max_steps,
    )

    assert result.terminal_cause == expected


def test_evaluate_checkpoint_hashes_validates_and_runs_unique_seeded_rollouts(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")
    model = SimpleNamespace()
    load_calls = []
    validation_calls = []

    class FakePPO:
        @staticmethod
        def load(path, *, device):
            load_calls.append((path, device))
            return model

    def fake_validate(actual_model, *, action_count, extractor_name):
        validation_calls.append((actual_model, action_count, extractor_name))

    def fake_rollout(actual_model, level, seed, deterministic, **_kwargs):
        assert actual_model is model
        return _rollout(level, seed, cleared=seed % 2 == 0, max_x=seed)

    monkeypatch.setattr(evaluation, "PPO", FakePPO)
    monkeypatch.setattr(evaluation, "validate_resume_model", fake_validate)
    monkeypatch.setattr(evaluation, "evaluate_rollout", fake_rollout)

    report = evaluate_checkpoint(
        checkpoint,
        levels=["1-1", "1-2"],
        episodes=2,
        seed=42000,
        deterministic=False,
    )

    assert load_calls == [(checkpoint, "cpu")]
    assert validation_calls == [(model, 12, "impala")]
    assert report.checkpoint_sha256 == hashlib.sha256(b"checkpoint").hexdigest()
    assert report.requested_episodes == 2
    assert report.deterministic is False
    assert [rollout.seed for rollout in report.rollouts] == [
        42000,
        42001,
        42002,
        42003,
    ]
    assert report.stages["1-1"]["episodes"] == 2
    assert report.stages["1-2"]["episodes"] == 2


def test_cli_resolves_all_writes_report_and_prints_32_rows(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "reports" / "evaluation.json"
    captured = {}

    def fake_evaluate_checkpoint(
        model_path: Path,
        levels,
        episodes=15,
        seed=42000,
        deterministic=False,
    ):
        captured.update(
            model_path=model_path,
            levels=tuple(levels),
            episodes=episodes,
            seed=seed,
            deterministic=deterministic,
        )
        return _report()

    monkeypatch.setattr(
        evaluation, "evaluate_checkpoint", fake_evaluate_checkpoint
    )

    report = evaluation.main(
        [
            "--model",
            "checkpoint.zip",
            "--levels",
            "all",
            "--episodes",
            "15",
            "--stochastic",
            "--seed",
            "42000",
            "--out",
            str(output_path),
        ]
    )

    assert captured == {
        "model_path": Path("checkpoint.zip"),
        "levels": ALL_LEVELS,
        "episodes": 15,
        "seed": 42000,
        "deterministic": False,
    }
    assert EvaluationReport.read(output_path) == report
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 34
    assert lines[-1] == "PASS"


def test_cli_deterministic_diagnostic_cannot_print_pass(
    tmp_path, monkeypatch, capsys
):
    report = replace(_report(), deterministic=True)
    monkeypatch.setattr(
        evaluation,
        "evaluate_checkpoint",
        lambda *_args, **_kwargs: report,
    )

    evaluation.main(
        [
            "--model",
            "checkpoint.zip",
            "--levels",
            "all",
            "--deterministic",
            "--out",
            str(tmp_path / "evaluation.json"),
        ]
    )

    assert capsys.readouterr().out.strip().splitlines()[-1] == "FAIL"


def test_cli_defaults_to_stochastic_mode():
    args = evaluation.parse_args(
        [
            "--model",
            "checkpoint.zip",
            "--levels",
            "1-1",
            "--out",
            "evaluation.json",
        ]
    )

    assert args.deterministic is False
