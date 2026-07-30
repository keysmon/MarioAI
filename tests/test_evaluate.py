from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import marioai.evaluate as evaluation
import marioai.results as results
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


def _acceptance_report() -> EvaluationReport:
    rollouts = []
    stages = {}
    for level_index, level in enumerate(ALL_LEVELS):
        stage_rollouts = [
            _rollout(
                level,
                42000 + level_index * 15 + episode_index,
                cleared=episode_index == 0,
            )
            for episode_index in range(15)
        ]
        rollouts.extend(stage_rollouts)
        stages[level] = summarize_stage(level, stage_rollouts)
    return EvaluationReport(
        checkpoint_sha256="a" * 64,
        deterministic=False,
        requested_episodes=15,
        stages=stages,
        rollouts=rollouts,
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
    report = _acceptance_report()

    assert report.passed is True
    assert len(report.rollouts) == 480
    assert not replace(
        report,
        stages={key: value for key, value in report.stages.items() if key != "8-4"},
    ).passed


def test_acceptance_rejects_deterministic_or_non_15_report():
    report = _acceptance_report()

    assert not replace(report, deterministic=True).passed
    assert not replace(report, requested_episodes=14).passed


def test_legacy_policy_report_is_explicitly_diagnostic_and_never_acceptance():
    report = replace(
        _acceptance_report(),
        policy_mode="legacy_simple_nature_diagnostic",
    )

    assert report.policy_mode == "legacy_simple_nature_diagnostic"
    assert not report.passed


def test_unlabeled_legacy_json_defaults_to_non_acceptance_mode():
    payload = _acceptance_report().to_dict()
    payload.pop("policy_mode")

    report = EvaluationReport.from_dict(payload)

    assert report.policy_mode == "legacy_simple_nature_diagnostic"
    assert not report.passed


def test_acceptance_rejects_malformed_policy_mode_type():
    report = replace(_acceptance_report(), policy_mode=[])

    assert not report.passed


def test_acceptance_rejects_stage_without_15_episodes_or_a_clear():
    report = _acceptance_report()
    wrong_count = dict(report.stages)
    wrong_count["8-4"] = {**wrong_count["8-4"], "episodes": 14}
    no_clear = dict(report.stages)
    no_clear["8-4"] = {**no_clear["8-4"], "passed": False, "clears": 0}

    assert not replace(report, stages=wrong_count).passed
    assert not replace(report, stages=no_clear).passed


def test_acceptance_rejects_missing_or_duplicate_rollout_seeds():
    report = _acceptance_report()
    duplicate_seed = replace(
        report.rollouts[-1],
        seed=report.rollouts[0].seed,
    )

    assert not replace(report, rollouts=report.rollouts[:-1]).passed
    assert not replace(
        report,
        rollouts=[*report.rollouts[:-1], duplicate_seed],
    ).passed


def test_acceptance_rejects_clear_terminal_cause_inconsistency():
    report = _acceptance_report()
    inconsistent = replace(report.rollouts[0], terminal_cause="death")

    assert not replace(
        report,
        rollouts=[inconsistent, *report.rollouts[1:]],
    ).passed


def test_acceptance_rejects_stage_summary_not_recomputed_from_rollouts():
    report = _acceptance_report()
    stages = dict(report.stages)
    stages["1-1"] = {
        **stages["1-1"],
        "clears": 2,
        "clear_rate": 2 / 15,
    }

    assert not replace(report, stages=stages).passed


@pytest.mark.parametrize(
    "checkpoint_sha256",
    [
        "a" * 63,
        "a" * 65,
        "g" * 64,
        123,
    ],
)
def test_acceptance_rejects_malformed_checkpoint_digest(checkpoint_sha256):
    assert not replace(
        _acceptance_report(), checkpoint_sha256=checkpoint_sha256
    ).passed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deterministic", 0),
        ("requested_episodes", 15.0),
        ("rollouts", ()),
        ("rollouts", [object()]),
    ],
)
def test_acceptance_rejects_malformed_top_level_field_types(field, value):
    assert not replace(_acceptance_report(), **{field: value}).passed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clears", True),
        ("clears", 1.0),
        ("clears", -1),
        ("clears", 16),
        ("episodes", 15.0),
        ("episodes", True),
    ],
)
def test_acceptance_requires_true_integer_stage_counts(field, value):
    report = _acceptance_report()
    stages = dict(report.stages)
    stages["8-4"] = {**stages["8-4"], field: value}

    assert not replace(report, stages=stages).passed


def test_acceptance_rejects_non_mapping_stage_summary():
    report = _acceptance_report()
    stages = dict(report.stages)
    stages["8-4"] = []

    assert not replace(report, stages=stages).passed


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


def test_checkpoint_progress_ordering_is_stable_across_representations():
    summary_only = _report(passed_stages=8, clears=12, mean_max_x=100.0)
    partially_rollout_backed = replace(
        summary_only,
        rollouts=[_rollout("1-1", seed=0, max_x=100)],
    )
    rollout_backed = replace(
        summary_only,
        rollouts=[
            _rollout(level, seed=index, max_x=100)
            for index, level in enumerate(ALL_LEVELS)
        ],
    )
    improved_rollouts = list(rollout_backed.rollouts)
    improved_rollouts[-1] = _rollout("8-4", seed=31, max_x=101)
    improved = replace(rollout_backed, rollouts=improved_rollouts)

    assert checkpoint_score(partially_rollout_backed) == checkpoint_score(
        summary_only
    )
    assert checkpoint_score(rollout_backed) == checkpoint_score(summary_only)
    assert is_better_checkpoint(improved, summary_only)


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
        "policy_mode": "shared_complex_impala",
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


def test_report_write_creates_destination_atomically(tmp_path, monkeypatch):
    report = _report()
    path = tmp_path / "evaluation.json"
    links = []
    original_link = results.os.link

    def recording_link(source, destination):
        links.append((Path(source), Path(destination)))
        return original_link(source, destination)

    monkeypatch.setattr(results.os, "link", recording_link)

    report.write(path)

    assert len(links) == 1
    temporary, destination = links[0]
    assert temporary.parent == path.parent
    assert temporary != path
    assert destination == path
    assert not temporary.exists()


def test_report_write_refuses_existing_path_without_explicit_overwrite(tmp_path):
    path = tmp_path / "evaluation.json"
    original = _report(passed_stages=1)
    replacement = _report(passed_stages=2)
    original.write(path)
    original_bytes = path.read_bytes()

    with pytest.raises(FileExistsError, match="overwrite"):
        replacement.write(path)

    assert path.read_bytes() == original_bytes


def test_report_write_explicit_overwrite_replaces_atomically(tmp_path):
    path = tmp_path / "evaluation.json"
    original = _report(passed_stages=1)
    replacement = _report(passed_stages=2)
    original.write(path)

    replacement.write(path, overwrite=True)

    assert EvaluationReport.read(path) == replacement


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


def test_rollout_stops_at_aware_deadline_and_closes_environment(monkeypatch):
    deadline = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
    times = iter(
        [
            deadline - timedelta(seconds=1),
            deadline - timedelta(seconds=1),
            deadline,
        ]
    )

    class MultiStepEnv(_OneStepVectorEnv):
        def __init__(self):
            super().__init__()
            self.steps = 0

        def step(self, action):
            self.steps += 1
            return (
                np.array([0]),
                np.array([0.0]),
                np.array([False]),
                [{"flag_get": False, "time": 200, "x_pos": int(action[0])}],
            )

    environment = MultiStepEnv()
    monkeypatch.setattr(
        evaluation,
        "_make_evaluation_env",
        lambda **_kwargs: environment,
    )

    with pytest.raises(
        evaluation.EvaluationDeadlineReached,
        match="deadline",
    ):
        evaluate_rollout(
            _SeededPolicy(),
            "1-1",
            seed=5,
            deterministic=False,
            deadline=deadline,
            now=lambda: next(times),
        )

    assert environment.steps == 1
    assert environment.closed is True


def test_evaluation_factory_pins_complex_action_set():
    environment = evaluation._make_evaluation_env(
        level="1-1",
        frame_stack=4,
        skip=4,
        shape=84,
        action_set="complex",
    )
    try:
        assert environment.action_space.n == 12
    finally:
        environment.close()


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

    def fake_validate(actual_model, **kwargs):
        validation_calls.append((actual_model, kwargs))

    def fake_rollout(actual_model, level, seed, deterministic, **kwargs):
        assert actual_model is model
        assert kwargs["action_set"] == "complex"
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
    assert validation_calls == [
        (
            model,
            {
                "action_count": 12,
                "observation_shape": (84, 84),
                "frame_stack": 4,
                "extractor_name": "impala",
                "features_dim": 512,
                "channels": (16, 32, 32),
            },
        )
    ]
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


def test_legacy_checkpoint_evaluation_uses_simple_nature_diagnostic_mode(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"legacy checkpoint")
    model = SimpleNamespace()
    validation_calls = []
    rollout_action_sets = []

    class FakePPO:
        @staticmethod
        def load(_path, *, device):
            assert device == "cpu"
            return model

    def fake_validate(actual_model, **kwargs):
        validation_calls.append((actual_model, kwargs))

    def fake_rollout(actual_model, level, seed, deterministic, **kwargs):
        assert actual_model is model
        rollout_action_sets.append(kwargs["action_set"])
        return _rollout(level, seed)

    monkeypatch.setattr(evaluation, "PPO", FakePPO)
    monkeypatch.setattr(evaluation, "validate_resume_model", fake_validate)
    monkeypatch.setattr(evaluation, "evaluate_rollout", fake_rollout)

    report = evaluate_checkpoint(
        checkpoint,
        levels=["1-1"],
        episodes=1,
        legacy_diagnostic=True,
    )

    assert report.policy_mode == "legacy_simple_nature_diagnostic"
    assert validation_calls == [
        (
            model,
            {
                "action_count": 7,
                "observation_shape": (84, 84),
                "frame_stack": 4,
                "extractor_name": "nature",
                "features_dim": None,
                "channels": None,
            },
        )
    ]
    assert rollout_action_sets == ["simple"]
    assert not report.passed


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
        legacy_diagnostic=False,
    ):
        captured.update(
            model_path=model_path,
            levels=tuple(levels),
            episodes=episodes,
            seed=seed,
            deterministic=deterministic,
            legacy_diagnostic=legacy_diagnostic,
        )
        return _acceptance_report()

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
        "legacy_diagnostic": False,
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


def test_cli_requires_explicit_overwrite_for_existing_report(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "evaluation.json"
    output_path.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(
        evaluation,
        "evaluate_checkpoint",
        lambda *_args, **_kwargs: _acceptance_report(),
    )
    argv = [
        "--model",
        "checkpoint.zip",
        "--levels",
        "all",
        "--out",
        str(output_path),
    ]

    with pytest.raises(FileExistsError, match="overwrite"):
        evaluation.main(argv)

    report = evaluation.main([*argv, "--overwrite"])

    assert EvaluationReport.read(output_path) == report


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
    assert args.legacy_diagnostic is False


def test_cli_legacy_diagnostic_requires_an_explicit_flag():
    args = evaluation.parse_args(
        [
            "--model",
            "checkpoint.zip",
            "--levels",
            "1-1",
            "--legacy-diagnostic",
            "--out",
            "evaluation.json",
        ]
    )

    assert args.legacy_diagnostic is True
