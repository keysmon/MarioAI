"""Immutable, durable accounting for paid training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping


class BudgetExceeded(ValueError):
    """Raised when an accrued or projected expense exceeds an approved limit."""


def _decimal(value: Any, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return value


def _nonnegative_decimal(value: Any, field_name: str) -> Decimal:
    value = _decimal(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return value


def _json_decimal(value: Any, field_name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be stored as a JSON string")
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field_name} is not a valid Decimal") from error


@dataclass(frozen=True)
class CostedRun:
    """The cumulative runtime and rates for one phase/instance pair."""

    phase: str
    instance_id: str
    hours: Decimal
    instance_hourly_usd: Decimal
    volume_hourly_usd: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("phase must be a non-empty string")
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be a non-empty string")
        _nonnegative_decimal(self.hours, "hours")
        _nonnegative_decimal(self.instance_hourly_usd, "instance_hourly_usd")
        _nonnegative_decimal(self.volume_hourly_usd, "volume_hourly_usd")

    @property
    def cost_usd(self) -> Decimal:
        return self.hours * (self.instance_hourly_usd + self.volume_hourly_usd)

    def to_dict(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "instance_id": self.instance_id,
            "hours": str(self.hours),
            "instance_hourly_usd": str(self.instance_hourly_usd),
            "volume_hourly_usd": str(self.volume_hourly_usd),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CostedRun:
        try:
            if not isinstance(value, Mapping):
                raise ValueError("costed run must be a mapping")
            return cls(
                phase=value["phase"],
                instance_id=value["instance_id"],
                hours=_json_decimal(value["hours"], "hours"),
                instance_hourly_usd=_json_decimal(
                    value["instance_hourly_usd"], "instance_hourly_usd"
                ),
                volume_hourly_usd=_json_decimal(
                    value["volume_hourly_usd"], "volume_hourly_usd"
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid costed run") from error


@dataclass(frozen=True)
class BudgetLedger:
    """A hard-capped ledger whose updates replace cumulative run progress."""

    cap_usd: Decimal = Decimal("50.00")
    spent_usd: Decimal = Decimal("0")
    runs: tuple[CostedRun, ...] = ()
    allocations: Mapping[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonnegative_decimal(self.cap_usd, "cap_usd")
        _nonnegative_decimal(self.spent_usd, "spent_usd")
        if self.spent_usd > self.cap_usd:
            raise BudgetExceeded("spent amount exceeds the hard cap")
        if not isinstance(self.runs, tuple) or not all(
            isinstance(run, CostedRun) for run in self.runs
        ):
            raise ValueError("runs must be a tuple of CostedRun values")
        if len({(run.phase, run.instance_id) for run in self.runs}) != len(self.runs):
            raise ValueError("runs must have unique phase/instance pairs")
        if self.spent_usd < sum((run.cost_usd for run in self.runs), Decimal("0")):
            raise ValueError("spent_usd cannot be below represented run costs")
        if not isinstance(self.allocations, Mapping):
            raise ValueError("allocations must be a mapping")
        normalized_allocations: dict[str, Decimal] = {}
        for phase, allocation in self.allocations.items():
            if not isinstance(phase, str) or not phase:
                raise ValueError("allocation phase must be a non-empty string")
            normalized_allocations[phase] = _nonnegative_decimal(
                allocation, "allocation"
            )
        object.__setattr__(
            self,
            "allocations",
            MappingProxyType(dict(sorted(normalized_allocations.items()))),
        )
        for phase, allocation in self.allocations.items():
            if self._phase_spent(phase) > allocation:
                raise BudgetExceeded(f"phase allocation exceeded for {phase}")

    @property
    def remaining_usd(self) -> Decimal:
        return self.cap_usd - self.spent_usd

    def _phase_spent(self, phase: str) -> Decimal:
        return sum((run.cost_usd for run in self.runs if run.phase == phase), Decimal("0"))

    def _enforce_phase_allocation(self, phase: str, spent_usd: Decimal) -> None:
        allocation = self.allocations.get(phase)
        if allocation is not None and spent_usd > allocation:
            raise BudgetExceeded(f"phase allocation exceeded for {phase}")

    def update_run(self, run: CostedRun) -> BudgetLedger:
        """Return a ledger with cumulative progress for ``run`` safely replaced."""
        if not isinstance(run, CostedRun):
            raise ValueError("run must be a CostedRun")
        key = (run.phase, run.instance_id)
        old_run = next(
            (existing for existing in self.runs if (existing.phase, existing.instance_id) == key),
            None,
        )
        if old_run is not None and run.hours < old_run.hours:
            raise ValueError("run hours cannot decrease")
        if old_run is not None and (
            run.instance_hourly_usd != old_run.instance_hourly_usd
            or run.volume_hourly_usd != old_run.volume_hourly_usd
        ):
            raise ValueError("run rates cannot change")
        replacement_cost = old_run.cost_usd if old_run is not None else Decimal("0")
        projected_spent = self.spent_usd - replacement_cost + run.cost_usd
        if projected_spent > self.cap_usd:
            raise BudgetExceeded("hard cap would be exceeded")
        projected_phase_spent = self._phase_spent(run.phase) - replacement_cost + run.cost_usd
        self._enforce_phase_allocation(run.phase, projected_phase_spent)
        updated_runs = tuple(
            run if (existing.phase, existing.instance_id) == key else existing
            for existing in self.runs
        )
        if old_run is None:
            updated_runs += (run,)
        return BudgetLedger(
            cap_usd=self.cap_usd,
            spent_usd=projected_spent,
            runs=updated_runs,
            allocations=self.allocations,
        )

    def require_launch(
        self,
        phase: str,
        hourly_usd: Decimal,
        volume_hourly_usd: Decimal,
        max_hours: Decimal,
        reserve_usd: Decimal,
    ) -> None:
        """Raise before launch when the worst-case cost cannot fit the ledger."""
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a non-empty string")
        hourly_usd = _nonnegative_decimal(hourly_usd, "hourly_usd")
        volume_hourly_usd = _nonnegative_decimal(
            volume_hourly_usd, "volume_hourly_usd"
        )
        max_hours = _nonnegative_decimal(max_hours, "max_hours")
        reserve_usd = _nonnegative_decimal(reserve_usd, "reserve_usd")
        projected_run_cost = max_hours * (hourly_usd + volume_hourly_usd)
        if self.spent_usd + projected_run_cost + reserve_usd > self.cap_usd:
            raise BudgetExceeded("hard cap would be exceeded")
        self._enforce_phase_allocation(
            phase, self._phase_spent(phase) + projected_run_cost
        )

    def save(self, path: Path) -> None:
        """Atomically persist Decimal values as strings beside the target path."""
        payload = {
            "cap_usd": str(self.cap_usd),
            "spent_usd": str(self.spent_usd),
            "runs": [run.to_dict() for run in self.runs],
            "allocations": {
                phase: str(allocation)
                for phase, allocation in self.allocations.items()
            },
        }
        path = Path(path)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            json.dump(payload, temporary, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(
        cls, path: Path, cap_usd: Decimal = Decimal("50.00")
    ) -> BudgetLedger:
        """Load a prior ledger, or return an empty ledger when none exists yet."""
        path = Path(path)
        if not path.exists():
            return cls(cap_usd=cap_usd)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("ledger payload must be a mapping")
            runs = payload["runs"]
            allocations = payload.get("allocations", {})
            if not isinstance(runs, list):
                raise ValueError("runs must be a list")
            if not isinstance(allocations, Mapping):
                raise ValueError("allocations must be a mapping")
            return cls(
                cap_usd=_json_decimal(payload["cap_usd"], "cap_usd"),
                spent_usd=_json_decimal(payload["spent_usd"], "spent_usd"),
                runs=tuple(CostedRun.from_dict(run) for run in runs),
                allocations={
                    phase: _json_decimal(allocation, "allocation")
                    for phase, allocation in allocations.items()
                },
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid budget ledger at {path}") from error
