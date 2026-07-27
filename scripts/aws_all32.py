#!/usr/bin/env python
"""Budget-guarded lifecycle orchestration for all-32 AWS training."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_CEILING
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

from marioai.aws import AwsCli, AwsConfig, PreflightResult, SpotOffer
from marioai.budget import BudgetLedger, CostedRun


_SECONDS_PER_HOUR = Decimal("3600")
_HOURS_PER_BILLING_MONTH = Decimal("720")
_PREFLIGHT_AUTH_TTL_SECONDS = Decimal("300")
_AMI_ID_PATTERN = re.compile(r"ami-[0-9a-f]{8,17}")
_CLIENT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
_INSTANCE_ID_PATTERN = re.compile(r"i-[0-9a-f]{8,17}")
_AWS_COMMAND_TIMEOUT_SECONDS = 60
_INSTANCE_READY_TIMEOUT_SECONDS = 600
_INSTANCE_READY_POLL_SECONDS = 5
_SSH_READY_TIMEOUT_SECONDS = 300
_SSH_READY_POLL_SECONDS = 5
_LAUNCH_STATE_SCHEMA_VERSION = 1


class AwsLifecycleError(RuntimeError):
    """Raised when a paid-instance lifecycle cannot be handled safely."""


@dataclass(frozen=True)
class LaunchReservation:
    """Durable pre-mutation identity and prices for one launch attempt."""

    client_token: str
    state: str
    phase: str
    request: dict[str, Any]
    instance_hourly_usd: Decimal
    on_demand_hourly_usd: Decimal
    volume_hourly_usd: Decimal
    max_hours: Decimal
    grace_hours: Decimal
    requested_epoch_seconds: Decimal
    instance_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _LAUNCH_STATE_SCHEMA_VERSION,
            "client_token": self.client_token,
            "state": self.state,
            "phase": self.phase,
            "request": self.request,
            "instance_hourly_usd": str(self.instance_hourly_usd),
            "on_demand_hourly_usd": str(self.on_demand_hourly_usd),
            "volume_hourly_usd": str(self.volume_hourly_usd),
            "max_hours": str(self.max_hours),
            "grace_hours": str(self.grace_hours),
            "requested_epoch_seconds": str(self.requested_epoch_seconds),
            "instance_id": self.instance_id,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> LaunchReservation:
        expected = {
            "schema_version",
            "client_token",
            "state",
            "phase",
            "request",
            "instance_hourly_usd",
            "on_demand_hourly_usd",
            "volume_hourly_usd",
            "max_hours",
            "grace_hours",
            "requested_epoch_seconds",
            "instance_id",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise AwsLifecycleError("launch reservation has an invalid schema")
        if payload["schema_version"] != _LAUNCH_STATE_SCHEMA_VERSION:
            raise AwsLifecycleError(
                "launch reservation has an unsupported schema version"
            )
        try:
            reservation = cls(
                client_token=payload["client_token"],
                state=payload["state"],
                phase=payload["phase"],
                request=payload["request"],
                instance_hourly_usd=Decimal(
                    payload["instance_hourly_usd"]
                ),
                on_demand_hourly_usd=Decimal(
                    payload["on_demand_hourly_usd"]
                ),
                volume_hourly_usd=Decimal(payload["volume_hourly_usd"]),
                max_hours=Decimal(payload["max_hours"]),
                grace_hours=Decimal(payload["grace_hours"]),
                requested_epoch_seconds=Decimal(
                    payload["requested_epoch_seconds"]
                ),
                instance_id=payload["instance_id"],
            )
        except (ArithmeticError, TypeError, ValueError) as error:
            raise AwsLifecycleError(
                "launch reservation contains invalid decimals"
            ) from error
        reservation._validate()
        return reservation

    def _validate(self) -> None:
        if (
            not isinstance(self.client_token, str)
            or _CLIENT_TOKEN_PATTERN.fullmatch(self.client_token) is None
        ):
            raise AwsLifecycleError(
                "launch reservation has an invalid ClientToken"
            )
        if not isinstance(self.phase, str) or not self.phase:
            raise AwsLifecycleError("launch reservation has an invalid phase")
        if self.state not in {"reserved", "launched"}:
            raise AwsLifecycleError("launch reservation has an invalid state")
        if not isinstance(self.request, dict):
            raise AwsLifecycleError("launch reservation has an invalid request")
        for name, value in (
            ("instance_hourly_usd", self.instance_hourly_usd),
            ("on_demand_hourly_usd", self.on_demand_hourly_usd),
            ("volume_hourly_usd", self.volume_hourly_usd),
            ("max_hours", self.max_hours),
            ("grace_hours", self.grace_hours),
            ("requested_epoch_seconds", self.requested_epoch_seconds),
        ):
            if not value.is_finite() or value < 0:
                raise AwsLifecycleError(
                    f"launch reservation has invalid {name}"
                )
        if self.max_hours <= 0:
            raise AwsLifecycleError(
                "launch reservation max_hours must be positive"
            )
        if self.instance_id is not None and (
            not isinstance(self.instance_id, str)
            or _INSTANCE_ID_PATTERN.fullmatch(self.instance_id) is None
        ):
            raise AwsLifecycleError(
                "launch reservation has an invalid instance ID"
            )
        if (self.state == "reserved") != (self.instance_id is None):
            raise AwsLifecycleError(
                "launch reservation state and instance ID disagree"
            )


class LaunchStateStore:
    """Serialize lifecycle access and atomically persist a launch reservation."""

    def __init__(
        self,
        ledger_path: Path,
        *,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.state_path = self.ledger_path.with_name(
            f"{self.ledger_path.name}.launch.json"
        )
        self.lock_path = self.ledger_path.with_name(
            f"{self.ledger_path.name}.lock"
        )
        self.wall_clock = wall_clock
        self._lock_file: Any = None

    def __enter__(self) -> LaunchStateStore:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            self._lock_file.close()
            self._lock_file = None
            raise AwsLifecycleError(
                f"another AWS lifecycle owns ledger {self.ledger_path}"
            ) from error
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def now_epoch_seconds(self) -> Decimal:
        value = self.wall_clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AwsLifecycleError("wall clock returned an invalid value")
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise AwsLifecycleError("wall clock returned an invalid value")
        return result

    def load(self) -> LaunchReservation | None:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AwsLifecycleError(
                f"cannot read launch reservation {self.state_path}"
            ) from error
        return LaunchReservation.from_dict(payload)

    def save(self, reservation: LaunchReservation) -> None:
        if not isinstance(reservation, LaunchReservation):
            raise ValueError("reservation must be a LaunchReservation")
        reservation._validate()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.state_path.parent,
            delete=False,
        ) as temporary:
            json.dump(reservation.to_dict(), temporary, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, self.state_path)
            directory_fd = os.open(self.state_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def clear(self) -> None:
        self.state_path.unlink(missing_ok=True)
        directory_fd = os.open(self.state_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


@dataclass(frozen=True)
class LaunchedInstance:
    """The immutable prices and identity used to account for one instance."""

    phase: str
    instance_id: str
    instance_type: str
    availability_zone: str
    subnet_id: str
    ami_id: str
    public_ip: str | None
    spot_hourly_usd: Decimal
    volume_hourly_usd: Decimal
    max_hours: Decimal
    launched_monotonic: Decimal


class AwsCommandAdapter:
    """Separate command boundary for the three Task 3 control operations."""

    def __init__(
        self,
        *,
        config: AwsConfig,
        readonly: AwsCli,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        if not isinstance(config, AwsConfig):
            raise ValueError("config must be an AwsConfig")
        profile = getattr(readonly, "profile", None)
        region = getattr(readonly, "region", None)
        if not isinstance(profile, str) or not profile:
            raise ValueError("readonly adapter must expose a profile")
        if not isinstance(region, str) or not region:
            raise ValueError("readonly adapter must expose a region")
        self.readonly = readonly
        self.config = config
        self.profile = profile
        self.region = region
        self._runner = runner
        self._authorized_subnet_azs: set[tuple[str, str]] = set()
        self._resolved_ami_id: str | None = None
        self._authorized_termination_ids: set[str] = set()

    def preflight(self, config: AwsConfig) -> PreflightResult:
        """Delegate only to the public read-only boundary."""
        self._authorized_subnet_azs.clear()
        result = self.readonly.preflight(config)
        if not isinstance(result, PreflightResult):
            raise AwsLifecycleError("preflight returned an invalid result")
        if config != self.config:
            raise AwsLifecycleError(
                "preflight configuration does not match lifecycle adapter"
            )
        self._authorized_subnet_azs.update(result.subnet_azs)
        return result

    def latest_spot_prices(
        self, instance_types: Any
    ) -> tuple[SpotOffer, ...]:
        """Delegate current offer lookup to the authorized read-only boundary."""
        return self.readonly.latest_spot_prices(instance_types)

    def verify_account(self, config: AwsConfig) -> None:
        """Verify the exact account before status or emergency mutation."""
        if self.profile != config.profile or self.region != config.region:
            raise AwsLifecycleError(
                "AWS CLI profile/region does not match configuration"
            )
        payload = self.readonly.run(["sts", "get-caller-identity"])
        if not isinstance(payload, dict) or payload.get(
            "Account"
        ) != config.account_id:
            raise AwsLifecycleError(
                f"expected AWS account {config.account_id} before lifecycle action"
            )

    def resolve_ami(self, parameter_name: str) -> str:
        """Resolve only the configured public SSM AMI parameter."""
        if parameter_name != self.config.ami_ssm_parameter or not (
            isinstance(parameter_name, str)
            and parameter_name.startswith("/aws/service/")
        ):
            raise AwsLifecycleError(
                "AMI parameter must be an AWS public SSM parameter"
            )
        payload = self._invoke(
            [
                "ssm",
                "get-parameter",
                "--name",
                parameter_name,
                "--query",
                "Parameter.Value",
            ]
        )
        if (
            not isinstance(payload, str)
            or _AMI_ID_PATTERN.fullmatch(payload) is None
        ):
            raise AwsLifecycleError("SSM get-parameter returned an invalid AMI")
        self._resolved_ami_id = payload
        return payload

    def run_instances(self, request: Any) -> dict[str, Any]:
        """Perform one validated, idempotent one-time Spot request."""
        _validate_launch_request(
            request,
            config=self.config,
            authorized_subnet_azs=self._authorized_subnet_azs,
            resolved_ami_id=self._resolved_ami_id,
        )
        request_json = json.dumps(
            request, sort_keys=True, separators=(",", ":")
        )
        try:
            payload = self._invoke(
                [
                    "ec2",
                    "run-instances",
                    "--cli-input-json",
                    request_json,
                ]
            )
        except subprocess.TimeoutExpired as error:
            raise AwsLifecycleError(
                "AWS ec2 run-instances timed out with an ambiguous result; "
                f"reconcile using ClientToken {request['ClientToken']!r} "
                "and do not submit a new token"
            ) from error
        if not isinstance(payload, dict):
            raise AwsLifecycleError(
                "AWS ec2 run-instances returned a non-object response"
            )
        instance = _launched_instance_payload(payload)
        self._authorized_termination_ids.add(instance["InstanceId"])
        return payload

    def terminate_instance(self, instance_id: str) -> None:
        """Idempotently request termination of one validated EC2 instance ID."""
        if (
            not isinstance(instance_id, str)
            or _INSTANCE_ID_PATTERN.fullmatch(instance_id) is None
        ):
            raise AwsLifecycleError("invalid EC2 instance ID")
        if instance_id not in self._authorized_termination_ids:
            raise AwsLifecycleError(
                "EC2 instance is not an authorized MarioAI-All32 target"
            )
        payload = self._invoke(
            [
                "ec2",
                "terminate-instances",
                "--instance-ids",
                instance_id,
            ]
        )
        if not isinstance(payload, dict):
            raise AwsLifecycleError(
                "AWS ec2 terminate-instances returned a non-object response"
            )
        _validate_termination_response(payload, instance_id)

    def project_instances(
        self,
        *,
        instance_id: str | None = None,
        client_token: str | None = None,
        active_only: bool = False,
    ) -> tuple[dict[str, str | None], ...]:
        """Return strict project instance summaries through the read-only CLI."""
        if instance_id is not None and (
            not isinstance(instance_id, str)
            or _INSTANCE_ID_PATTERN.fullmatch(instance_id) is None
        ):
            raise AwsLifecycleError("invalid EC2 instance ID")
        if client_token is not None and (
            not isinstance(client_token, str)
            or _CLIENT_TOKEN_PATTERN.fullmatch(client_token) is None
        ):
            raise AwsLifecycleError("invalid EC2 ClientToken")
        if instance_id is not None and client_token is not None:
            raise AwsLifecycleError(
                "instance ID and ClientToken filters are mutually exclusive"
            )
        args = ["ec2", "describe-instances"]
        if instance_id is not None:
            args.extend(["--instance-ids", instance_id])
        args.extend(
            [
                "--filters",
                "Name=tag:Project,Values=MarioAI-All32",
            ]
        )
        if client_token is not None:
            args.append(f"Name=client-token,Values={client_token}")
        if instance_id is None:
            if active_only:
                args.append(
                    "Name=instance-state-name,"
                    "Values=pending,running,stopping,stopped,shutting-down"
                )
        payload = self.readonly.run(args)
        summaries = _instance_summaries(payload)
        self._authorized_termination_ids.update(
            summary["instance_id"]
            for summary in summaries
            if isinstance(summary["instance_id"], str)
        )
        return summaries

    def _invoke(self, args: list[str]) -> dict | list | str:
        command = [
            "aws",
            *args,
            "--profile",
            self.profile,
            "--region",
            self.region,
            "--output",
            "json",
        ]
        try:
            completed = self._runner(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=_AWS_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise AwsLifecycleError(
                f"AWS {' '.join(args[:2])} failed{detail}"
            ) from error
        except OSError as error:
            raise AwsLifecycleError(
                f"could not execute AWS CLI for {' '.join(args[:2])}: {error}"
            ) from error
        stdout = completed.stdout
        if not isinstance(stdout, str):
            raise AwsLifecycleError(
                f"AWS {' '.join(args[:2])} returned non-text output"
            )
        if not stdout.strip():
            return ""
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise AwsLifecycleError(
                f"AWS {' '.join(args[:2])} returned invalid JSON"
            ) from error
        if not isinstance(payload, (dict, list, str)):
            raise AwsLifecycleError(
                f"AWS {' '.join(args[:2])} returned unsupported JSON"
            )
        return payload


@dataclass(frozen=True)
class RemotePoll:
    """One successful remote/instance liveness observation."""

    running: bool
    exit_code: int | None = None


class SshRemoteSupervisor:
    """Upload the repository and start the independent remote shell guard."""

    def __init__(
        self,
        *,
        aws: Any,
        ssh_key: Path,
        local_repo: Path,
        runner: Callable[..., Any] = subprocess.run,
        remote_repo: str = "/home/ubuntu/MarioAI",
        user: str = "ubuntu",
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        startup_confirmation_seconds: int = 10,
    ) -> None:
        self.aws = aws
        self.ssh_key = Path(ssh_key)
        self.local_repo = Path(local_repo)
        self._runner = runner
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._startup_confirmation_seconds = startup_confirmation_seconds
        self._hosts: dict[str, str] = {}
        if not self.ssh_key.is_file():
            raise AwsLifecycleError(f"SSH key does not exist: {self.ssh_key}")
        if not self.local_repo.is_dir():
            raise AwsLifecycleError(
                f"local repository does not exist: {self.local_repo}"
            )
        if re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_repo) is None:
            raise AwsLifecycleError("remote repository path is unsafe")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", user) is None:
            raise AwsLifecycleError("SSH user is unsafe")
        if (
            isinstance(startup_confirmation_seconds, bool)
            or not isinstance(startup_confirmation_seconds, int)
            or startup_confirmation_seconds <= 0
        ):
            raise AwsLifecycleError(
                "startup confirmation seconds must be positive"
            )
        self.remote_repo = remote_repo.rstrip("/")
        self.user = user

    def start(
        self,
        instance: LaunchedInstance,
        *,
        phase: str,
        max_seconds: int,
        s3_prefix: str,
    ) -> None:
        """Rsync source and launch cloud_train.sh under a recorded remote PID."""
        host = _validated_public_ip(instance.public_ip)
        self._hosts[instance.instance_id] = host
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", phase) is None:
            raise AwsLifecycleError("phase is unsafe for remote execution")
        if (
            isinstance(max_seconds, bool)
            or not isinstance(max_seconds, int)
            or max_seconds <= 0
        ):
            raise AwsLifecycleError("max_seconds must be a positive integer")
        if (
            not isinstance(s3_prefix, str)
            or re.fullmatch(r"s3://[^/]+/.+/", s3_prefix) is None
        ):
            raise AwsLifecycleError("S3 prefix is invalid")
        target = f"{self.user}@{host}"
        self._wait_for_ssh(target)
        os_bootstrap = """\
set -eu
. /etc/os-release
[ "$ID" = ubuntu ]
[ "$VERSION_ID" = 24.04 ]
sudo env DEBIAN_FRONTEND=noninteractive apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  --no-install-recommends ca-certificates curl unzip rsync python3 \
  python3-venv
if ! command -v aws >/dev/null 2>&1; then
  work_dir="$(mktemp -d)"
  trap 'rm -rf "$work_dir"' EXIT
  curl --fail --location --silent --show-error \
    https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip \
    --output "$work_dir/awscliv2.zip"
  unzip -q "$work_dir/awscliv2.zip" -d "$work_dir"
  sudo "$work_dir/aws/install" --install-dir /usr/local/aws-cli \
    --bin-dir /usr/local/bin
fi
command -v rsync
aws --version
"""
        self._run_remote_command(
            [*self._ssh_command(target), "bash", "-s"],
            timeout=900,
            input_text=os_bootstrap,
        )
        ssh_transport = (
            f"ssh -i {shlex.quote(str(self.ssh_key))} "
            "-o BatchMode=yes -o StrictHostKeyChecking=accept-new "
            "-o ConnectTimeout=10 -o ServerAliveInterval=15 "
            "-o ServerAliveCountMax=2"
        )
        self._run_remote_command(
            [
                "rsync",
                "--archive",
                "--compress",
                "--exclude",
                ".git",
                "--exclude",
                ".venv",
                "--exclude",
                "__pycache__",
                "--rsh",
                ssh_transport,
                f"{self.local_repo}/",
                f"{target}:{self.remote_repo}/",
            ],
            timeout=300,
        )
        dependency_bootstrap = """\
set -eu
repo=$1
bootstrap="$HOME/.marioai-bootstrap"
python3 -m venv "$bootstrap"
"$bootstrap/bin/python" -m pip install \
  --disable-pip-version-check --upgrade pip uv
"$bootstrap/bin/uv" python install 3.13
"$bootstrap/bin/uv" venv --clear --python 3.13 "$repo/.venv"
"$bootstrap/bin/uv" pip install --python "$repo/.venv/bin/python" \
  -r "$repo/requirements.txt" --editable "$repo"
mkdir -p "$repo/models" "$repo/reports"
"$repo/.venv/bin/python" -c 'import marioai, torch'
aws sts get-caller-identity --output json >/dev/null
"""
        self._run_remote_command(
            [
                *self._ssh_command(target),
                "bash",
                "-s",
                "--",
                shlex.quote(self.remote_repo),
            ],
            timeout=1800,
            input_text=dependency_bootstrap,
        )
        elapsed_seconds = int(
            max(
                Decimal("0"),
                _monotonic_decimal(self._monotonic())
                - instance.launched_monotonic,
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        remaining_seconds = max_seconds - elapsed_seconds
        if remaining_seconds <= 0:
            raise AwsLifecycleError(
                "maximum paid runtime expired during remote bootstrap"
            )
        startup_script = (
            "set -eu\n"
            "repo=$1\n"
            "shift\n"
            'cd "$repo"\n'
            "pidfile=/tmp/marioai-cloud-train.pid\n"
            "logfile=/tmp/marioai-cloud-train.log\n"
            'rm -f "$pidfile"\n'
            "nohup ./scripts/cloud_train.sh \"$@\" "
            "</dev/null >\"$logfile\" 2>&1 &\n"
            "pid=$!\n"
            'tmp_pidfile="${pidfile}.$$"\n'
            "printf '%s\\n' \"$pid\" >\"$tmp_pidfile\"\n"
            'mv "$tmp_pidfile" "$pidfile"\n'
            f"remaining={self._startup_confirmation_seconds}\n"
            'while [ "$remaining" -gt 0 ]; do\n'
            "  sleep 1\n"
            '  if ! kill -0 "$pid" 2>/dev/null; then\n'
            "    set +e\n"
            '    wait "$pid"\n'
            "    status=$?\n"
            "    set -e\n"
            '    tail -n 40 "$logfile" >&2 || true\n'
            '    if [ "$status" -eq 0 ]; then exit 70; fi\n'
            '    exit "$status"\n'
            "  fi\n"
            '  remaining=$((remaining - 1))\n'
            "done\n"
        )
        remote_args = [
            self.remote_repo,
            phase,
            str(remaining_seconds),
            self.remote_repo,
            s3_prefix,
        ]
        self._run_remote_command(
            [
                *self._ssh_command(target),
                "bash",
                "-s",
                "--",
                *[shlex.quote(argument) for argument in remote_args],
            ],
            timeout=max(30, self._startup_confirmation_seconds + 20),
            input_text=startup_script,
        )

    def poll(self, instance: LaunchedInstance) -> RemotePoll:
        summaries = self.aws.project_instances(
            instance_id=instance.instance_id
        )
        if not summaries:
            return RemotePoll(running=False, exit_code=0)
        state = summaries[0]["state"]
        return RemotePoll(
            running=state in {"pending", "running"},
            exit_code=None if state in {"pending", "running"} else 0,
        )

    def request_shutdown(self, instance: LaunchedInstance) -> None:
        host = self._hosts.get(instance.instance_id)
        if host is None:
            host = _validated_public_ip(instance.public_ip)
        target = f"{self.user}@{host}"
        shutdown_script = (
            "set -eu\n"
            "if [ -r /tmp/marioai-cloud-train.pid ]; then\n"
            "  pid=\"$(cat /tmp/marioai-cloud-train.pid)\"\n"
            "  kill -TERM \"$pid\"\n"
            "  remaining=180\n"
            "  while kill -0 \"$pid\" 2>/dev/null; do\n"
            "    if [ \"$remaining\" -le 0 ]; then exit 124; fi\n"
            "    sleep 5\n"
            "    remaining=$((remaining - 1))\n"
            "  done\n"
            "else\n"
            "  sudo shutdown -h now\n"
            "fi\n"
        )
        self._run_remote_command(
            [
                *self._ssh_command(target),
                "bash",
                "-s",
            ],
            timeout=900,
            input_text=shutdown_script,
        )

    def _ssh_command(self, target: str) -> list[str]:
        return [
            "ssh",
            "-i",
            str(self.ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            target,
        ]

    def _wait_for_ssh(self, target: str) -> None:
        deadline = (
            _monotonic_decimal(self._monotonic())
            + Decimal(_SSH_READY_TIMEOUT_SECONDS)
        )
        last_detail = ""
        while True:
            try:
                self._runner(
                    [*self._ssh_command(target), "bash", "-s"],
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=15,
                    input="exit 0\n",
                )
                return
            except subprocess.CalledProcessError as error:
                last_detail = (error.stderr or "").strip()
            except subprocess.TimeoutExpired:
                last_detail = "SSH attempt timed out"
            except OSError as error:
                last_detail = str(error)
            remaining = deadline - _monotonic_decimal(self._monotonic())
            if remaining <= 0:
                suffix = f": {last_detail}" if last_detail else ""
                raise AwsLifecycleError(
                    "SSH did not become ready within 300 seconds" + suffix
                )
            self._sleeper(
                float(min(Decimal(_SSH_READY_POLL_SECONDS), remaining))
            )

    def _run_remote_command(
        self,
        command: list[str],
        *,
        timeout: int,
        input_text: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "check": True,
            "text": True,
            "capture_output": True,
            "timeout": timeout,
        }
        if input_text is not None:
            kwargs["input"] = input_text
        try:
            self._runner(command, **kwargs)
        except subprocess.TimeoutExpired as error:
            raise AwsLifecycleError(
                f"{command[0]} timed out after {timeout} seconds"
            ) from error
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise AwsLifecycleError(f"{command[0]} failed{detail}") from error
        except OSError as error:
            raise AwsLifecycleError(
                f"could not execute {command[0]}: {error}"
            ) from error


class AwsOrchestrator:
    """Coordinate guarded launch, accounting, and unconditional termination."""

    def __init__(
        self,
        *,
        config: AwsConfig,
        aws: Any,
        ledger: BudgetLedger,
        remote: Any,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        client_token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        reservation_store: LaunchStateStore | None = None,
    ) -> None:
        if not isinstance(config, AwsConfig):
            raise ValueError("config must be an AwsConfig")
        if not isinstance(ledger, BudgetLedger):
            raise ValueError("ledger must be a BudgetLedger")
        self.config = config
        self.aws = aws
        self.ledger = ledger
        self.remote = remote
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._client_token_factory = client_token_factory
        self.reservation_store = reservation_store
        self._preflight_result: PreflightResult | None = None
        self._preflight_authorized_at: Decimal | None = None

    def preflight(self) -> PreflightResult:
        """Run the exact read-only preflight before authorizing this process."""
        self._preflight_result = None
        self._preflight_authorized_at = None
        result = self.aws.preflight(self.config)
        if not isinstance(result, PreflightResult):
            raise AwsLifecycleError("preflight returned an invalid result")
        self._preflight_result = result
        self._preflight_authorized_at = _monotonic_decimal(self._monotonic())
        return result

    @property
    def volume_hourly_usd(self) -> Decimal:
        """Return a conservative 30-day-month gp3 hourly rate."""
        return (
            self.config.gp3_monthly_usd_per_gb
            * Decimal(self.config.root_volume_gb)
            / _HOURS_PER_BILLING_MONTH
        )

    def launch_guarded_instance(
        self, phase: str, max_hours: Decimal
    ) -> LaunchedInstance:
        """Launch one allowed Spot instance after the conservative budget gate."""
        if (
            isinstance(self.aws, AwsCommandAdapter)
            and self.reservation_store is None
        ):
            raise AwsLifecycleError(
                "real AWS launch requires a locked durable reservation store"
            )
        self._require_fresh_preflight()
        if phase not in self.config.allocations:
            raise ValueError(f"phase {phase!r} is not configured")
        if (
            not isinstance(max_hours, Decimal)
            or not max_hours.is_finite()
            or max_hours <= 0
        ):
            raise ValueError("max_hours must be a positive finite Decimal")

        offers = self.aws.latest_spot_prices(self.config.instance_types)
        if (
            not isinstance(offers, tuple)
            or not offers
            or not all(isinstance(offer, SpotOffer) for offer in offers)
        ):
            raise AwsLifecycleError("no valid allowed Spot offer is available")
        offer = min(
            offers,
            key=lambda item: (
                item.hourly_usd,
                item.instance_type,
                item.availability_zone,
                item.subnet_id,
            ),
        )
        if (
            offer.instance_type not in self.config.instance_types
            or (
                offer.subnet_id,
                offer.availability_zone,
            )
            not in set(self._preflight_result.subnet_azs)
        ):
            raise AwsLifecycleError(
                "Spot offer is outside the preflight-authorized configuration"
            )
        on_demand_hourly = self.config.on_demand_ceiling_usd[
            offer.instance_type
        ]
        grace_hours = Decimal(self.config.grace_minutes) / Decimal("60")
        reserve_usd = (
            on_demand_hourly + self.volume_hourly_usd
        ) * grace_hours
        self.ledger.require_launch(
            phase,
            on_demand_hourly,
            self.volume_hourly_usd,
            max_hours,
            reserve_usd,
        )

        ami_id = self.aws.resolve_ami(self.config.ami_ssm_parameter)
        if (
            not isinstance(ami_id, str)
            or _AMI_ID_PATTERN.fullmatch(ami_id) is None
        ):
            raise AwsLifecycleError(
                "SSM returned an invalid AMI ID; launch was not attempted"
            )
        client_token = self._client_token_factory()
        if (
            not isinstance(client_token, str)
            or _CLIENT_TOKEN_PATTERN.fullmatch(client_token) is None
        ):
            raise AwsLifecycleError(
                "client token must contain 1-64 safe characters"
            )
        self._require_fresh_preflight()
        tags = [
            {"Key": "Project", "Value": "MarioAI-All32"},
            {"Key": "Phase", "Value": phase},
        ]
        request = {
            "ImageId": ami_id,
            "InstanceType": offer.instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "ClientToken": client_token,
            "UserData": _shutdown_user_data(max_hours),
            "KeyName": self.config.key_name,
            "IamInstanceProfile": {"Name": self.config.instance_profile},
            "Placement": {"AvailabilityZone": offer.availability_zone},
            "InstanceMarketOptions": {
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time"},
            },
            "InstanceInitiatedShutdownBehavior": "terminate",
            "NetworkInterfaces": [
                {
                    "AssociatePublicIpAddress": True,
                    "DeleteOnTermination": True,
                    "DeviceIndex": 0,
                    "Groups": [self.config.security_group_id],
                    "SubnetId": offer.subnet_id,
                }
            ],
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "DeleteOnTermination": True,
                        "Encrypted": True,
                        "VolumeSize": self.config.root_volume_gb,
                        "VolumeType": "gp3",
                    },
                }
            ],
            "TagSpecifications": [
                {"ResourceType": "instance", "Tags": tags},
                {"ResourceType": "volume", "Tags": tags},
            ],
        }
        reservation = LaunchReservation(
            client_token=client_token,
            state="reserved",
            phase=phase,
            request=request,
            instance_hourly_usd=offer.hourly_usd,
            on_demand_hourly_usd=on_demand_hourly,
            volume_hourly_usd=self.volume_hourly_usd,
            max_hours=max_hours,
            grace_hours=grace_hours,
            requested_epoch_seconds=(
                self.reservation_store.now_epoch_seconds()
                if self.reservation_store is not None
                else Decimal("0")
            ),
        )
        if self.reservation_store is not None:
            if self.reservation_store.load() is not None:
                raise AwsLifecycleError(
                    "an unresolved launch reservation exists; run reconcile"
                )
            self.reservation_store.save(reservation)
        self._preflight_result = None
        self._preflight_authorized_at = None
        launch_requested_at = _monotonic_decimal(self._monotonic())
        payload = self.aws.run_instances(request)
        try:
            instance = _launched_instance_payload(payload)
        except AwsLifecycleError as error:
            raise AwsLifecycleError(
                f"{error}; reconcile the ambiguous launch using "
                f"ClientToken {client_token!r}"
            ) from error
        if self.reservation_store is not None:
            self.reservation_store.save(
                replace(
                    reservation,
                    state="launched",
                    instance_id=instance["InstanceId"],
                )
            )
        return LaunchedInstance(
            phase=phase,
            instance_id=instance["InstanceId"],
            instance_type=offer.instance_type,
            availability_zone=offer.availability_zone,
            subnet_id=offer.subnet_id,
            ami_id=ami_id,
            public_ip=instance.get("PublicIpAddress"),
            spot_hourly_usd=offer.hourly_usd,
            volume_hourly_usd=self.volume_hourly_usd,
            max_hours=max_hours,
            launched_monotonic=launch_requested_at,
        )

    def _require_fresh_preflight(self) -> None:
        authorized_at = self._preflight_authorized_at
        if self._preflight_result is None or authorized_at is None:
            raise AwsLifecycleError(
                "a successful preflight is required before launch"
            )
        now = _monotonic_decimal(self._monotonic())
        age = now - authorized_at
        if age < 0 or age > _PREFLIGHT_AUTH_TTL_SECONDS:
            self._preflight_result = None
            self._preflight_authorized_at = None
            raise AwsLifecycleError(
                "preflight authorization expired; run preflight again"
            )

    def wait_for_running_public_ip(
        self, instance: LaunchedInstance
    ) -> LaunchedInstance:
        """Wait boundedly for the launched instance and its public IPv4."""
        deadline = (
            _monotonic_decimal(self._monotonic())
            + Decimal(_INSTANCE_READY_TIMEOUT_SECONDS)
        )
        while True:
            summaries = self.aws.project_instances(
                instance_id=instance.instance_id
            )
            if not isinstance(summaries, tuple) or len(summaries) > 1:
                raise AwsLifecycleError(
                    "describe-instances returned duplicate launch targets"
                )
            if summaries:
                summary = summaries[0]
                if summary.get("instance_id") != instance.instance_id:
                    raise AwsLifecycleError(
                        "describe-instances returned a different launch target"
                    )
                state = summary.get("state")
                if state in {
                    "stopping",
                    "stopped",
                    "shutting-down",
                    "terminated",
                }:
                    raise AwsLifecycleError(
                        f"instance entered {state!r} before remote startup"
                    )
                if state not in {"pending", "running"}:
                    raise AwsLifecycleError(
                        f"instance returned unexpected state {state!r}"
                    )
                public_ip = summary.get("public_ip")
                if state == "running" and public_ip is not None:
                    return replace(
                        instance,
                        public_ip=_validated_public_ip(public_ip),
                    )
            now = _monotonic_decimal(self._monotonic())
            remaining = deadline - now
            if remaining <= 0:
                raise AwsLifecycleError(
                    "instance did not become running with a public IP "
                    "within 600 seconds"
                )
            self._sleeper(
                float(
                    min(
                        Decimal(_INSTANCE_READY_POLL_SECONDS),
                        remaining,
                    )
                )
            )

    def persist_elapsed(
        self, instance: LaunchedInstance, ledger_path: Path
    ) -> CostedRun:
        """Reload and durably persist monotonic observed cost."""
        ledger_path = Path(ledger_path)
        if ledger_path.exists():
            self.ledger = _configured_ledger(ledger_path, self.config)
        elapsed_seconds = max(
            Decimal("0"),
            _monotonic_decimal(self._monotonic())
            - instance.launched_monotonic,
        )
        prior_run = next(
            (
                prior
                for prior in self.ledger.runs
                if prior.phase == instance.phase
                and prior.instance_id == instance.instance_id
            ),
            None,
        )
        elapsed_hours = elapsed_seconds / _SECONDS_PER_HOUR
        if prior_run is not None:
            elapsed_hours = max(elapsed_hours, prior_run.hours)
        run = CostedRun(
            phase=instance.phase,
            instance_id=instance.instance_id,
            hours=elapsed_hours,
            instance_hourly_usd=instance.spot_hourly_usd,
            volume_hourly_usd=instance.volume_hourly_usd,
        )
        self.ledger = self.ledger.update_run(run)
        self.ledger.save(ledger_path)
        return run

    def terminate_and_settle(
        self, instance: LaunchedInstance, ledger_path: Path
    ) -> CostedRun:
        """Terminate, wait boundedly for terminal state, and persist final cost."""
        termination_error: BaseException | None = None
        terminal_confirmed = False
        try:
            self.aws.terminate_instance(instance.instance_id)
            deadline = (
                _monotonic_decimal(self._monotonic())
                + Decimal(self.config.grace_minutes * 60)
            )
            while True:
                summaries = self.aws.project_instances(
                    instance_id=instance.instance_id
                )
                self.persist_elapsed(instance, ledger_path)
                if not summaries or summaries[0].get("state") == "terminated":
                    terminal_confirmed = True
                    break
                if len(summaries) != 1 or summaries[0].get(
                    "instance_id"
                ) != instance.instance_id:
                    raise AwsLifecycleError(
                        "termination status returned an unexpected target"
                    )
                remaining = deadline - _monotonic_decimal(self._monotonic())
                if remaining <= 0:
                    raise AwsLifecycleError(
                        "instance did not reach terminated state within "
                        f"{self.config.grace_minutes} minutes"
                    )
                self._sleeper(float(min(Decimal("60"), remaining)))
        except BaseException as error:
            termination_error = error

        accounting_error: BaseException | None = None
        try:
            final_run = self.persist_elapsed(instance, ledger_path)
        except BaseException as error:
            accounting_error = error
            final_run = None

        if (
            terminal_confirmed
            and accounting_error is None
            and self.reservation_store is not None
        ):
            self.reservation_store.clear()
        if termination_error is not None:
            if accounting_error is not None:
                termination_error.add_note(
                    "final cost persistence also failed: "
                    f"{type(accounting_error).__name__}: {accounting_error}"
                )
            raise termination_error
        if accounting_error is not None:
            raise accounting_error
        assert final_run is not None
        return final_run

    def monitor_and_terminate(
        self, instance: LaunchedInstance, ledger_path: Path
    ) -> CostedRun:
        """Persist every successful poll and always terminate the EC2 instance."""
        if not isinstance(instance, LaunchedInstance):
            raise ValueError("instance must be a LaunchedInstance")
        body_error: BaseException | None = None
        final_run: CostedRun | None = None
        try:
            while True:
                poll = self.remote.poll(instance)
                running = getattr(poll, "running", None)
                if not isinstance(running, bool):
                    raise AwsLifecycleError(
                        "remote poll returned an invalid running state"
                    )
                run = self.persist_elapsed(instance, ledger_path)
                if not running:
                    break
                if (
                    self.ledger.spent_usd
                    >= self.config.shutdown_threshold_usd
                    or run.hours >= instance.max_hours
                ):
                    self.remote.request_shutdown(instance)
                    break
                self._sleeper(60)
        except BaseException as error:
            body_error = error
        try:
            final_run = self.terminate_and_settle(instance, ledger_path)
        except BaseException as termination_error:
            if body_error is None:
                raise
            body_error.add_note(
                "EC2 termination also failed (including final accounting): "
                f"{type(termination_error).__name__}: {termination_error}"
            )
        if body_error is not None:
            raise body_error.with_traceback(body_error.__traceback__)
        assert final_run is not None
        return final_run


def _monotonic_decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AwsLifecycleError("monotonic clock returned an invalid value")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise AwsLifecycleError("monotonic clock returned an invalid value")
    return result


def _validated_public_ip(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AwsLifecycleError(
            "instance has no public IP address for remote supervision"
        )
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise AwsLifecycleError(
            "instance returned an invalid public IP address"
        ) from error
    if address.version != 4:
        raise AwsLifecycleError("remote supervision requires a public IPv4 address")
    return value


def _launched_instance_payload(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise AwsLifecycleError("run-instances response must be a JSON object")
    instances = payload.get("Instances")
    if not isinstance(instances, list) or len(instances) != 1:
        raise AwsLifecycleError(
            "run-instances response must contain exactly one instance"
        )
    instance = instances[0]
    if not isinstance(instance, dict):
        raise AwsLifecycleError("run-instances returned a malformed instance")
    instance_id = instance.get("InstanceId")
    if (
        not isinstance(instance_id, str)
        or _INSTANCE_ID_PATTERN.fullmatch(instance_id) is None
    ):
        raise AwsLifecycleError(
            "run-instances response has no valid instance ID"
        )
    public_ip = instance.get("PublicIpAddress")
    if public_ip is not None and (
        not isinstance(public_ip, str) or not public_ip
    ):
        raise AwsLifecycleError(
            "run-instances response has an invalid public IP address"
        )
    return instance


def _shutdown_user_data(max_hours: Decimal) -> str:
    seconds = int(
        (max_hours * _SECONDS_PER_HOUR).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    minutes = max(1, (seconds + 59) // 60)
    script = (
        "#!/bin/sh\n"
        f"shutdown -h +{minutes} 'MarioAI maximum paid runtime reached'\n"
    )
    return base64.b64encode(script.encode("utf-8")).decode("ascii")


def _validate_termination_response(
    payload: dict[str, Any], instance_id: str
) -> None:
    transitions = payload.get("TerminatingInstances")
    if not isinstance(transitions, list) or len(transitions) != 1:
        raise AwsLifecycleError(
            "termination response is ambiguous; run reconcile"
        )
    transition = transitions[0]
    current = (
        transition.get("CurrentState")
        if isinstance(transition, dict)
        else None
    )
    if (
        not isinstance(transition, dict)
        or transition.get("InstanceId") != instance_id
        or not isinstance(current, dict)
        or current.get("Name") not in {"shutting-down", "terminated"}
    ):
        raise AwsLifecycleError(
            "termination response is ambiguous; run reconcile"
        )


def _validate_launch_request(
    request: Any,
    *,
    config: AwsConfig,
    authorized_subnet_azs: set[tuple[str, str]],
    resolved_ami_id: str | None,
) -> None:
    if not isinstance(request, dict):
        raise AwsLifecycleError("run-instances request must be a mapping")
    expected_keys = {
        "ImageId",
        "InstanceType",
        "MinCount",
        "MaxCount",
        "ClientToken",
        "UserData",
        "KeyName",
        "IamInstanceProfile",
        "Placement",
        "InstanceMarketOptions",
        "InstanceInitiatedShutdownBehavior",
        "NetworkInterfaces",
        "BlockDeviceMappings",
        "TagSpecifications",
    }
    if set(request) != expected_keys:
        raise AwsLifecycleError(
            "run-instances request fields do not match the exact allowlist"
        )
    if (
        resolved_ami_id is None
        or request.get("ImageId") != resolved_ami_id
        or _AMI_ID_PATTERN.fullmatch(resolved_ami_id) is None
    ):
        raise AwsLifecycleError(
            "run-instances AMI does not match the resolved public parameter"
        )
    if request.get("InstanceType") not in config.instance_types:
        raise AwsLifecycleError(
            "run-instances instance type is outside configuration"
        )
    if request.get("KeyName") != config.key_name:
        raise AwsLifecycleError("run-instances key does not match configuration")
    if request.get("IamInstanceProfile") != {
        "Name": config.instance_profile
    }:
        raise AwsLifecycleError(
            "run-instances profile does not match configuration"
        )
    placement = request.get("Placement")
    if (
        not isinstance(placement, dict)
        or set(placement) != {"AvailabilityZone"}
        or not isinstance(placement["AvailabilityZone"], str)
    ):
        raise AwsLifecycleError("run-instances placement is invalid")
    market_options = request.get("InstanceMarketOptions")
    spot_options = (
        market_options.get("SpotOptions")
        if isinstance(market_options, dict)
        else None
    )
    if (
        not isinstance(market_options, dict)
        or set(market_options) != {"MarketType", "SpotOptions"}
        or market_options.get("MarketType") != "spot"
        or not isinstance(spot_options, dict)
        or set(spot_options) != {"SpotInstanceType"}
        or spot_options.get("SpotInstanceType") != "one-time"
    ):
        raise AwsLifecycleError(
            "run-instances requires a one-time Spot market request"
        )
    if (
        isinstance(request.get("MinCount"), bool)
        or isinstance(request.get("MaxCount"), bool)
        or request.get("MinCount") != 1
        or request.get("MaxCount") != 1
    ):
        raise AwsLifecycleError("run-instances must request exactly one instance")
    if request.get("InstanceInitiatedShutdownBehavior") != "terminate":
        raise AwsLifecycleError(
            "instance-initiated shutdown behavior must be terminate"
        )
    client_token = request.get("ClientToken")
    if (
        not isinstance(client_token, str)
        or _CLIENT_TOKEN_PATTERN.fullmatch(client_token) is None
    ):
        raise AwsLifecycleError(
            "run-instances requires a valid idempotency ClientToken"
        )
    network_interfaces = request.get("NetworkInterfaces")
    network = (
        network_interfaces[0]
        if isinstance(network_interfaces, list)
        and len(network_interfaces) == 1
        and isinstance(network_interfaces[0], dict)
        else None
    )
    if (
        not isinstance(network, dict)
        or network
        != {
            "AssociatePublicIpAddress": True,
            "DeleteOnTermination": True,
            "DeviceIndex": 0,
            "Groups": [config.security_group_id],
            "SubnetId": network.get("SubnetId"),
        }
        or network.get("SubnetId") not in config.subnet_ids
        or (
            network["SubnetId"],
            placement["AvailabilityZone"],
        )
        not in authorized_subnet_azs
    ):
        raise AwsLifecycleError(
            "run-instances requires one public network interface"
        )
    block_devices = request.get("BlockDeviceMappings")
    ebs = (
        block_devices[0].get("Ebs")
        if isinstance(block_devices, list)
        and len(block_devices) == 1
        and isinstance(block_devices[0], dict)
        else None
    )
    if (
        not isinstance(ebs, dict)
        or block_devices[0].get("DeviceName") != "/dev/sda1"
        or ebs
        != {
            "DeleteOnTermination": True,
            "Encrypted": True,
            "VolumeSize": config.root_volume_gb,
            "VolumeType": "gp3",
        }
    ):
        raise AwsLifecycleError(
            "run-instances requires encrypted delete-on-termination gp3"
        )
    tags = [
        {"Key": "Project", "Value": "MarioAI-All32"},
        {
            "Key": "Phase",
            "Value": next(
                (
                    item.get("Value")
                    for specification in request.get(
                        "TagSpecifications", []
                    )
                    if isinstance(specification, dict)
                    for item in specification.get("Tags", [])
                    if isinstance(item, dict)
                    and item.get("Key") == "Phase"
                ),
                None,
            ),
        },
    ]
    phase = tags[1]["Value"]
    if phase not in config.allocations or request.get(
        "TagSpecifications"
    ) != [
        {"ResourceType": "instance", "Tags": tags},
        {"ResourceType": "volume", "Tags": tags},
    ]:
        raise AwsLifecycleError(
            "run-instances tags do not match the configured project phase"
        )
    user_data = request.get("UserData")
    try:
        decoded_user_data = base64.b64decode(
            user_data, validate=True
        ).decode("utf-8")
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise AwsLifecycleError(
            "run-instances UserData is not valid base64"
        ) from error
    if re.fullmatch(
        r"#!/bin/sh\nshutdown -h \+[1-9][0-9]* "
        r"'MarioAI maximum paid runtime reached'\n",
        decoded_user_data,
    ) is None:
        raise AwsLifecycleError(
            "run-instances UserData lacks the shutdown guard"
        )


def _positive_decimal_argument(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except (ValueError, ArithmeticError) as error:
        raise argparse.ArgumentTypeError(
            "must be a positive decimal value"
        ) from error
    if not result.is_finite() or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive decimal value")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the five explicit lifecycle subcommands."""
    parser = argparse.ArgumentParser(
        description="Guarded AWS lifecycle for MarioAI all-32 training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="run exact read-only prerequisite checks"
    )
    preflight.add_argument("--config", required=True, type=Path)

    launch = subparsers.add_parser(
        "launch", help="preflight and launch one guarded Spot instance"
    )
    launch.add_argument("--config", required=True, type=Path)
    launch.add_argument("--ledger", required=True, type=Path)
    launch.add_argument("--phase", required=True)
    launch.add_argument(
        "--max-hours", required=True, type=_positive_decimal_argument
    )
    launch.add_argument("--ssh-key", type=Path)
    launch.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    status = subparsers.add_parser(
        "status", help="show Project=MarioAI-All32 instance state"
    )
    status.add_argument("--config", required=True, type=Path)
    status.add_argument("--instance-id")

    terminate = subparsers.add_parser(
        "terminate", help="idempotently terminate one instance"
    )
    terminate.add_argument("--config", required=True, type=Path)
    terminate.add_argument("--instance-id", required=True)

    reconcile = subparsers.add_parser(
        "reconcile", help="terminate active project instances and show ledger"
    )
    reconcile.add_argument("--config", required=True, type=Path)
    reconcile.add_argument("--ledger", required=True, type=Path)

    return parser


def _configured_ledger(path: Path, config: AwsConfig) -> BudgetLedger:
    path = Path(path)
    if not path.exists():
        return BudgetLedger(
            cap_usd=config.cap_usd, allocations=config.allocations
        )
    ledger = BudgetLedger.load(path, cap_usd=config.cap_usd)
    if ledger.cap_usd != config.cap_usd:
        raise AwsLifecycleError(
            "persisted ledger cap does not match the approved configuration"
        )
    if dict(ledger.allocations) != dict(config.allocations):
        raise AwsLifecycleError(
            "persisted ledger allocations do not match the approved configuration"
        )
    return ledger


def _settle_reservation(
    ledger: BudgetLedger,
    reservation: LaunchReservation,
    *,
    instance_id: str | None,
) -> BudgetLedger:
    """Consume the conservatively gated main runtime and global grace."""
    main_instance_id = instance_id or f"pending:{reservation.client_token}"
    prior_main = next(
        (
            run
            for run in ledger.runs
            if run.phase == reservation.phase
            and run.instance_id == main_instance_id
        ),
        None,
    )
    if prior_main is None:
        main_run = CostedRun(
            phase=reservation.phase,
            instance_id=main_instance_id,
            hours=reservation.max_hours,
            instance_hourly_usd=reservation.on_demand_hourly_usd,
            volume_hourly_usd=reservation.volume_hourly_usd,
        )
    else:
        reserved_main_cost = reservation.max_hours * (
            reservation.on_demand_hourly_usd
            + reservation.volume_hourly_usd
        )
        rate = (
            prior_main.instance_hourly_usd
            + prior_main.volume_hourly_usd
        )
        required_hours = (
            reserved_main_cost / rate if rate > 0 else reservation.max_hours
        )
        main_run = replace(
            prior_main,
            hours=max(prior_main.hours, required_hours),
        )
    settled = ledger.update_run(main_run)
    grace_run = CostedRun(
        phase="__launch_grace__",
        instance_id=f"pending:{reservation.client_token}",
        hours=reservation.grace_hours,
        instance_hourly_usd=reservation.on_demand_hourly_usd,
        volume_hourly_usd=reservation.volume_hourly_usd,
    )
    return settled.update_run(grace_run)


def _instance_summaries(payload: Any) -> tuple[dict[str, str | None], ...]:
    if not isinstance(payload, dict):
        raise AwsLifecycleError(
            "describe-instances response must be a JSON object"
        )
    reservations = payload.get("Reservations")
    if not isinstance(reservations, list):
        raise AwsLifecycleError(
            "describe-instances response has invalid Reservations"
        )
    summaries: list[dict[str, str | None]] = []
    for reservation in reservations:
        if not isinstance(reservation, dict):
            raise AwsLifecycleError(
                "describe-instances response has malformed reservation"
            )
        instances = reservation.get("Instances")
        if not isinstance(instances, list):
            raise AwsLifecycleError(
                "describe-instances response has invalid Instances"
            )
        for instance in instances:
            if not isinstance(instance, dict):
                raise AwsLifecycleError(
                    "describe-instances response has malformed instance"
                )
            instance_id = instance.get("InstanceId")
            instance_type = instance.get("InstanceType")
            state_payload = instance.get("State")
            state = (
                state_payload.get("Name")
                if isinstance(state_payload, dict)
                else None
            )
            public_ip = instance.get("PublicIpAddress")
            if (
                not isinstance(instance_id, str)
                or _INSTANCE_ID_PATTERN.fullmatch(instance_id) is None
                or not isinstance(instance_type, str)
                or not instance_type
                or not isinstance(state, str)
                or not state
                or (
                    public_ip is not None
                    and (not isinstance(public_ip, str) or not public_ip)
                )
            ):
                raise AwsLifecycleError(
                    "describe-instances returned an invalid instance summary"
                )
            summaries.append(
                {
                    "instance_id": instance_id,
                    "instance_type": instance_type,
                    "public_ip": public_ip,
                    "state": state,
                }
            )
    return tuple(
        sorted(summaries, key=lambda item: str(item["instance_id"]))
    )


def _write_json(stream: Any, payload: Any) -> None:
    json.dump(payload, stream, sort_keys=True)
    stream.write("\n")


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Any] = subprocess.run,
    stdout: Any = None,
    clock: Callable[[], Any] | None = None,
    aws_override: Any = None,
    remote: Any = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    client_token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    wall_clock: Callable[[], float] = time.time,
) -> int:
    """Dispatch one explicit lifecycle command."""
    args = build_parser().parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    config = AwsConfig.from_yaml(args.config)
    cli_kwargs: dict[str, Any] = {
        "profile": config.profile,
        "region": config.region,
        "runner": runner,
    }
    if clock is not None:
        cli_kwargs["clock"] = clock
    if aws_override is None:
        readonly = AwsCli(**cli_kwargs)
        aws = AwsCommandAdapter(
            config=config, readonly=readonly, runner=runner
        )
    else:
        aws = aws_override

    if args.command == "preflight":
        _write_json(output, asdict(aws.preflight(config)))
        return 0
    if args.command == "status":
        aws.verify_account(config)
        instances = aws.project_instances(instance_id=args.instance_id)
        _write_json(output, {"instances": list(instances)})
        return 0
    if args.command == "terminate":
        aws.verify_account(config)
        instances = aws.project_instances(instance_id=args.instance_id)
        if len(instances) > 1:
            raise AwsLifecycleError(
                "describe-instances returned duplicate project targets"
            )
        termination_requested = bool(instances)
        if termination_requested:
            instance_id = instances[0]["instance_id"]
            if instance_id != args.instance_id:
                raise AwsLifecycleError(
                    "describe-instances returned a different project target"
                )
            aws.terminate_instance(args.instance_id)
        _write_json(
            output,
            {
                "instance_id": args.instance_id,
                "termination_requested": termination_requested,
            },
        )
        return 0
    if args.command == "reconcile":
        with LaunchStateStore(
            args.ledger, wall_clock=wall_clock
        ) as reservation_store:
            aws.verify_account(config)
            ledger = _configured_ledger(args.ledger, config)
            reservation = reservation_store.load()
            instances = list(aws.project_instances(active_only=True))
            if reservation is not None:
                recovered = aws.project_instances(
                    client_token=reservation.client_token
                )
                known_ids = {
                    item["instance_id"]
                    for item in instances
                    if isinstance(item["instance_id"], str)
                }
                instances.extend(
                    item
                    for item in recovered
                    if item["instance_id"] not in known_ids
                )
            terminated_ids = []
            for instance in instances:
                instance_id = instance["instance_id"]
                if not isinstance(instance_id, str):
                    raise AwsLifecycleError(
                        "active instance summary has no valid instance ID"
                    )
                aws.terminate_instance(instance_id)
                terminated_ids.append(instance_id)
            reservation_cleared = reservation is None
            if reservation is not None:
                recovered_id = reservation.instance_id
                if recovered_id is None and instances:
                    candidate = instances[0]["instance_id"]
                    if isinstance(candidate, str):
                        recovered_id = candidate
                ledger = _settle_reservation(
                    ledger,
                    reservation,
                    instance_id=recovered_id,
                )
                ledger.save(args.ledger)
                remaining = aws.project_instances(active_only=True)
                if recovered_id is not None and not remaining:
                    reservation_store.clear()
                    reservation_cleared = True
            _write_json(
                output,
                {
                    "cap_usd": str(ledger.cap_usd),
                    "remaining_usd": str(ledger.remaining_usd),
                    "reservation_cleared": reservation_cleared,
                    "spent_usd": str(ledger.spent_usd),
                    "terminated_instance_ids": terminated_ids,
                },
            )
        return 0
    if args.command == "launch":
        with LaunchStateStore(
            args.ledger, wall_clock=wall_clock
        ) as reservation_store:
            if reservation_store.load() is not None:
                raise AwsLifecycleError(
                    "an unresolved launch reservation exists; run reconcile"
                )
            ledger = _configured_ledger(args.ledger, config)
            selected_remote = remote
            if selected_remote is None:
                ssh_key = args.ssh_key
                if ssh_key is None:
                    ssh_key = (
                        Path.home() / ".ssh" / f"{config.key_name}.pem"
                    )
                selected_remote = SshRemoteSupervisor(
                    aws=aws,
                    ssh_key=ssh_key,
                    local_repo=args.repo_dir,
                    runner=runner,
                    monotonic=monotonic,
                    sleeper=sleeper,
                )
            orchestrator = AwsOrchestrator(
                config=config,
                aws=aws,
                ledger=ledger,
                remote=selected_remote,
                monotonic=monotonic,
                sleeper=sleeper,
                client_token_factory=client_token_factory,
                reservation_store=reservation_store,
            )
            orchestrator.preflight()
            active_instances = aws.project_instances(active_only=True)
            if active_instances:
                raise AwsLifecycleError(
                    "active MarioAI-All32 instance exists; run reconcile"
                )
            instance = orchestrator.launch_guarded_instance(
                args.phase, args.max_hours
            )
            max_seconds = int(
                (args.max_hours * _SECONDS_PER_HOUR).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            try:
                orchestrator.persist_elapsed(instance, args.ledger)
                instance = orchestrator.wait_for_running_public_ip(instance)
                selected_remote.start(
                    instance,
                    phase=args.phase,
                    max_seconds=max_seconds,
                    s3_prefix=config.s3_prefix,
                )
            except BaseException as start_error:
                try:
                    orchestrator.terminate_and_settle(
                        instance, args.ledger
                    )
                except BaseException as termination_error:
                    start_error.add_note(
                        "EC2 termination/final accounting also failed: "
                        f"{type(termination_error).__name__}: "
                        f"{termination_error}"
                    )
                raise
            run = orchestrator.monitor_and_terminate(
                instance, args.ledger
            )
            _write_json(
                output,
                {
                    "costed_run": run.to_dict(),
                    "instance_id": instance.instance_id,
                },
            )
        return 0
    raise AwsLifecycleError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AwsLifecycleError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
