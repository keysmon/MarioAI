from dataclasses import replace
from decimal import Decimal
import json

import pytest

from marioai.budget import BudgetExceeded, BudgetLedger, CostedRun


@pytest.fixture
def run() -> CostedRun:
    return CostedRun(
        phase="benchmark",
        instance_id="i-test",
        hours=Decimal("1.0"),
        instance_hourly_usd=Decimal("0.60"),
        volume_hourly_usd=Decimal("0.011"),
    )


def test_accrual_uses_decimal_and_includes_volume_time():
    """Catches accounting that drops volume time or converts costs to floats."""
    ledger = BudgetLedger(cap_usd=Decimal("50.00"))

    updated = ledger.update_run(
        CostedRun(
            phase="benchmark",
            instance_id="i-test",
            hours=Decimal("2.5"),
            instance_hourly_usd=Decimal("0.60"),
            volume_hourly_usd=Decimal("0.011"),
        )
    )

    assert updated.spent_usd == Decimal("1.5275")


def test_launch_refused_before_projected_total_crosses_cap():
    """Catches a launch check that ignores expected runtime or reserve."""
    ledger = BudgetLedger(cap_usd=Decimal("50.00"), spent_usd=Decimal("47.00"))

    with pytest.raises(BudgetExceeded, match="hard cap"):
        ledger.require_launch(
            "evaluation",
            Decimal("0.60"),
            Decimal("0.011"),
            Decimal("4"),
            Decimal("1.00"),
        )


def test_atomic_roundtrip_preserves_runs_and_decimal_allocations(tmp_path, run):
    """Catches persistence that loses runs, allocations, or Decimal precision."""
    path = tmp_path / "spend.json"
    ledger = BudgetLedger(
        allocations={"benchmark": Decimal("4.00")}
    ).update_run(run)

    ledger.save(path)

    assert BudgetLedger.load(path) == ledger
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["allocations"] == {"benchmark": "4.00"}
    assert payload["runs"][0]["hours"] == "1.0"


def test_progress_update_is_idempotent_and_monotonic(run):
    """Catches double-counted progress and a replacement that permits rollback."""
    ledger = BudgetLedger()
    first = ledger.update_run(replace(run, hours=Decimal("1.0")))
    second = first.update_run(replace(run, hours=Decimal("1.5")))

    assert len(second.runs) == 1
    assert second.runs[0].hours == Decimal("1.5")
    assert second.spent_usd == Decimal("0.9165")
    with pytest.raises(ValueError, match="cannot decrease"):
        second.update_run(replace(run, hours=Decimal("1.0")))


def test_phase_allocation_blocks_accrual_and_projected_launch(run):
    """Catches phase allocations being bypassed despite a global budget balance."""
    ledger = BudgetLedger(allocations={"benchmark": Decimal("1.00")})
    accrued = ledger.update_run(replace(run, hours=Decimal("1.5")))

    with pytest.raises(BudgetExceeded, match="phase allocation"):
        accrued.update_run(replace(run, hours=Decimal("2.0")))
    with pytest.raises(BudgetExceeded, match="phase allocation"):
        accrued.require_launch(
            "benchmark",
            Decimal("0.60"),
            Decimal("0.011"),
            Decimal("0.2"),
            Decimal("0"),
        )


@pytest.mark.parametrize(
    "allocations",
    [
        {"benchmark": Decimal("-0.01")},
        {"benchmark": Decimal("NaN")},
        {"benchmark": "4.00"},
        {"": Decimal("4.00")},
    ],
)
def test_ledger_rejects_malformed_or_negative_allocations(allocations):
    """Catches accepting unsafe allocation data before it reaches the ledger."""
    with pytest.raises(ValueError):
        BudgetLedger(allocations=allocations)


def test_run_rejects_negative_duration():
    """Catches a run whose elapsed time would subtract accrued spend."""
    with pytest.raises(ValueError, match="hours cannot be negative"):
        CostedRun(
            phase="benchmark",
            instance_id="i-test",
            hours=Decimal("-0.1"),
            instance_hourly_usd=Decimal("0.60"),
            volume_hourly_usd=Decimal("0.011"),
        )


def test_progress_update_rejects_rate_change_without_restoring_spend(run):
    """Catches an update that rewrites already accrued hours at a lower rate."""
    first = BudgetLedger().update_run(run)

    with pytest.raises(ValueError, match="rates cannot change"):
        first.update_run(
            replace(
                run,
                hours=Decimal("2.0"),
                instance_hourly_usd=Decimal("0.01"),
            )
        )

    assert first.spent_usd == Decimal("0.6110")


def test_ledger_rejects_spend_below_represented_runs_and_allows_unattributed_spend(
    run,
):
    """Catches durable state that underreports already represented run costs."""
    with pytest.raises(ValueError, match="below represented run costs"):
        BudgetLedger(spent_usd=Decimal("0.60"), runs=(run,))

    ledger = BudgetLedger(spent_usd=Decimal("2.00"), runs=(run,))
    assert ledger.remaining_usd == Decimal("48.00")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("cap_usd", 50.0),
        lambda payload: payload.__setitem__("spent_usd", 0.0),
        lambda payload: payload["runs"][0].__setitem__("hours", 1.0),
        lambda payload: payload["runs"][0].__setitem__("instance_hourly_usd", 0.6),
        lambda payload: payload["allocations"].__setitem__("benchmark", 4.0),
        lambda payload: payload.__setitem__("allocations", []),
        lambda payload: payload["runs"][0].__setitem__("hours", "not-a-decimal"),
    ],
)
def test_load_rejects_non_string_or_malformed_persisted_schema(tmp_path, run, mutation):
    """Catches accepting numeric or malformed data in string-backed JSON fields."""
    path = tmp_path / "spend.json"
    payload = {
        "cap_usd": "50.00",
        "spent_usd": "0.6110",
        "runs": [run.to_dict()],
        "allocations": {"benchmark": "4.00"},
    }
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid budget ledger"):
        BudgetLedger.load(path)
