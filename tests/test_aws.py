from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

import marioai.train as training
from marioai.aws import (
    AwsCli,
    AwsCliError,
    AwsConfig,
    AwsPreflightError,
    PreflightResult,
    SpotOffer,
)
from marioai.budget import BudgetExceeded, BudgetLedger
from scripts.aws_all32 import AwsLifecycleError, AwsOrchestrator


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "aws-all32.yaml"
CLOUD_TRAIN_PATH = Path(__file__).parents[1] / "scripts" / "cloud_train.sh"
NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
SPOT_MAX_AGE = timedelta(days=7)
SPOT_MAX_FUTURE_SKEW = timedelta(minutes=5)
INSTANCE_PROFILE_ID = "AIPATESTMARIOALL32PROFILE"
SUBNET_IDS = (
    "subnet-0ba0242531d6615f3",
    "subnet-0870eed3fd2b7dfab",
    "subnet-02e9c5f8deb69ad62",
    "subnet-0e5d82120140b0b4e",
    "subnet-02ab8427ae66e6e36",
    "subnet-0fa9ec503414b2be9",
)


def _instance_profile_arn(config: AwsConfig) -> str:
    return (
        f"arn:aws:iam::{config.account_id}:instance-profile/"
        f"{config.instance_profile}"
    )


class FakeRunner:
    """Argument-vector fake; no process or network is ever started."""

    def __init__(self) -> None:
        self.responses: dict[tuple[str, ...], object] = {}
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.error: Exception | None = None

    def add(self, args: list[str], payload: object) -> None:
        self.responses[tuple(args)] = payload

    def __call__(self, command: list[str], **kwargs: object) -> SimpleNamespace:
        self.calls.append((command, kwargs))
        if self.error is not None:
            raise self.error
        operation_args = command[1:-6]
        payload = self.responses[tuple(operation_args)]
        return SimpleNamespace(stdout=json.dumps(payload), stderr="", returncode=0)


class FakeMonotonic:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeLifecycleAws:
    """In-memory lifecycle boundary; no AWS command is ever started."""

    def __init__(self, config: AwsConfig) -> None:
        self.config = config
        self.profile = config.profile
        self.region = config.region
        self.last_run_instances_request: dict[str, object] | None = None
        self.terminated_ids: list[str] = []
        self.ami_id = "ami-0123456789abcdef0"
        self.instance_profile_arn = _instance_profile_arn(config)
        self.instance_profile_id = INSTANCE_PROFILE_ID
        self.run_instances_response: object = {
            "Instances": [
                {
                    "InstanceId": "i-0123456789abcdef0",
                    "ClientToken": "task-3-idempotency-token",
                    "ImageId": "ami-0123456789abcdef0",
                    "InstanceType": "c7i.8xlarge",
                    "InstanceLifecycle": "spot",
                    "KeyName": config.key_name,
                    "IamInstanceProfile": {
                        "Arn": self.instance_profile_arn,
                        "Id": self.instance_profile_id,
                    },
                    "Placement": {"AvailabilityZone": "us-east-1a"},
                    "SubnetId": config.subnet_ids[0],
                    "SecurityGroups": [
                        {"GroupId": config.security_group_id}
                    ],
                    "Tags": [
                        {"Key": "Project", "Value": "MarioAI-All32"},
                        {"Key": "Phase", "Value": "benchmark"},
                    ],
                    "PublicIpAddress": "203.0.113.10",
                }
            ]
        }
        self._default_run_instances_response = self.run_instances_response
        self.offers: tuple[SpotOffer, ...] = (
            SpotOffer(
                instance_type="c7i.8xlarge",
                availability_zone="us-east-1a",
                subnet_id=self.config.subnet_ids[0],
                hourly_usd=Decimal("0.5568"),
                timestamp=NOW,
            ),
        )
        self.termination_error: BaseException | None = None
        self.run_instances_error: BaseException | None = None
        self.run_instances_hook = None
        self.resolve_ami_hook = None
        self.terminate_hook = None
        self.project_results: list[tuple[dict[str, object], ...]] = []
        self.preflight_calls = 0

    @property
    def run_instances_called(self) -> bool:
        return self.last_run_instances_request is not None

    def preflight(self, config: AwsConfig) -> PreflightResult:
        self.preflight_calls += 1
        return PreflightResult(
            account_id=config.account_id,
            vpc_id=config.vpc_id,
            subnet_azs=tuple(
                (subnet_id, f"{config.region}{suffix}")
                for subnet_id, suffix in zip(
                    config.subnet_ids, "abcdef", strict=True
                )
            ),
            s3_bucket="defectlens-phase3-002559670021",
            s3_key_prefix="marioai/all32/",
            running_project_instance_ids=(),
            instance_profile_arn=self.instance_profile_arn,
            instance_profile_id=self.instance_profile_id,
        )

    def latest_spot_prices(self, instance_types) -> tuple[SpotOffer, ...]:
        assert tuple(instance_types) == self.config.instance_types
        return self.offers

    def resolve_ami(self, parameter_name: str) -> str:
        assert parameter_name == self.config.ami_ssm_parameter
        if self.resolve_ami_hook is not None:
            self.resolve_ami_hook()
        return self.ami_id

    def run_instances(
        self,
        request: dict[str, object],
        *,
        max_hours: Decimal | None = None,
    ) -> object:
        del max_hours
        self.last_run_instances_request = request
        if self.run_instances_hook is not None:
            self.run_instances_hook()
        if self.run_instances_error is not None:
            raise self.run_instances_error
        if self.run_instances_response is self._default_run_instances_response:
            response_instance = self.run_instances_response["Instances"][0]
            response_instance.update(
                {
                    "ClientToken": request["ClientToken"],
                    "ImageId": request["ImageId"],
                    "InstanceType": request["InstanceType"],
                    "KeyName": request["KeyName"],
                    "IamInstanceProfile": {
                        "Arn": self.instance_profile_arn,
                        "Id": self.instance_profile_id,
                    },
                    "Placement": request["Placement"],
                    "SubnetId": request["NetworkInterfaces"][0]["SubnetId"],
                    "SecurityGroups": [
                        {
                            "GroupId": request["NetworkInterfaces"][0][
                                "Groups"
                            ][0]
                        }
                    ],
                    "Tags": request["TagSpecifications"][0]["Tags"],
                }
            )
        return self.run_instances_response

    def terminate_instance(self, instance_id: str) -> None:
        self.terminated_ids.append(instance_id)
        if self.terminate_hook is not None:
            self.terminate_hook()
        if self.termination_error is not None:
            raise self.termination_error

    def verify_account(self, config: AwsConfig) -> None:
        assert config == self.config

    def project_instances(
        self,
        *,
        instance_id: str | None = None,
        client_token: str | None = None,
        active_only: bool = False,
        timeout_seconds: int | float | None = None,
    ):
        del active_only, timeout_seconds
        if self.project_results:
            return self.project_results.pop(0)
        launched = self.run_instances_response
        if (
            not self.run_instances_called
            or not isinstance(launched, dict)
            or not isinstance(launched.get("Instances"), list)
            or not launched["Instances"]
        ):
            return ()
        payload = launched["Instances"][0]
        if not isinstance(payload, dict):
            return ()
        launched_id = payload.get("InstanceId")
        if (
            not isinstance(launched_id, str)
            or launched_id in self.terminated_ids
            or (instance_id is not None and instance_id != launched_id)
            or (
                client_token is not None
                and self.last_run_instances_request is not None
                and client_token
                != self.last_run_instances_request.get("ClientToken")
            )
        ):
            return ()
        return (
            {
                "instance_id": launched_id,
                "instance_type": self.last_run_instances_request[
                    "InstanceType"
                ],
                "public_ip": payload.get(
                    "PublicIpAddress", "203.0.113.10"
                ),
                "state": "running",
            },
        )


class FakeRemote:
    def __init__(self, clock: FakeMonotonic | None = None) -> None:
        self.exit_code = 0
        self.shutdown_requests: list[str] = []
        self.shutdown_deadlines: list[Decimal | None] = []
        self.poll_results: list[SimpleNamespace] = []
        self.poll_error: BaseException | None = None
        self.poll_times: list[float] = []
        self.clock = clock
        self.started: list[tuple[str, str, int, str]] = []
        self.training_args: list[tuple[str, ...]] = []
        self.start_error: BaseException | None = None

    def start(
        self,
        instance,
        *,
        phase: str,
        max_seconds: int,
        s3_prefix: str,
        train_args=(),
        on_tick=None,
        absolute_deadline=None,
    ) -> None:
        del on_tick, absolute_deadline
        if self.start_error is not None:
            raise self.start_error
        self.started.append(
            (instance.instance_id, phase, max_seconds, s3_prefix)
        )
        self.training_args.append(tuple(train_args))

    def poll(self, instance) -> SimpleNamespace:
        if self.clock is not None:
            self.poll_times.append(self.clock.now)
        if self.poll_error is not None:
            raise self.poll_error
        if self.poll_results:
            return self.poll_results.pop(0)
        return SimpleNamespace(running=False, exit_code=self.exit_code)

    def request_shutdown(
        self, instance, *, on_tick=None, absolute_deadline=None
    ) -> None:
        del on_tick
        self.shutdown_requests.append(instance.instance_id)
        self.shutdown_deadlines.append(absolute_deadline)


@pytest.fixture
def config() -> AwsConfig:
    return AwsConfig.from_yaml(CONFIG_PATH)


@pytest.fixture
def orchestrator(config: AwsConfig) -> AwsOrchestrator:
    return _make_orchestrator(config)


def _make_orchestrator(
    config: AwsConfig,
    *,
    ledger: BudgetLedger | None = None,
) -> AwsOrchestrator:
    clock = FakeMonotonic()
    remote = FakeRemote(clock)

    def sleep_and_advance(seconds: float) -> None:
        clock.now += seconds

    orchestrator = AwsOrchestrator(
        config=config,
        aws=FakeLifecycleAws(config),
        ledger=ledger
        or BudgetLedger(
            cap_usd=config.cap_usd, allocations=config.allocations
        ),
        remote=remote,
        monotonic=clock,
        sleeper=sleep_and_advance,
        client_token_factory=lambda: "task-3-idempotency-token",
    )
    orchestrator.preflight()
    orchestrator.test_clock = clock
    return orchestrator


def _subnet_payload(config: AwsConfig) -> dict[str, object]:
    return {
        "Subnets": [
            {
                "SubnetId": subnet_id,
                "VpcId": config.vpc_id,
                "AvailabilityZone": f"{config.region}{suffix}",
                "State": "available",
            }
            for subnet_id, suffix in zip(config.subnet_ids, "abcdef", strict=True)
        ]
    }


def _successful_runner(config: AwsConfig) -> FakeRunner:
    runner = FakeRunner()
    runner.add(
        ["sts", "get-caller-identity"],
        {
            "UserId": "AIDATEST",
            "Account": config.account_id,
            "Arn": f"arn:aws:iam::{config.account_id}:user/test",
        },
    )
    runner.add(
        ["ec2", "describe-vpcs", "--vpc-ids", config.vpc_id],
        {"Vpcs": [{"VpcId": config.vpc_id, "State": "available"}]},
    )
    runner.add(
        ["ec2", "describe-subnets", "--subnet-ids", *config.subnet_ids],
        _subnet_payload(config),
    )
    runner.add(
        [
            "ec2",
            "describe-security-groups",
            "--group-ids",
            config.security_group_id,
        ],
        {
            "SecurityGroups": [
                {
                    "GroupId": config.security_group_id,
                    "VpcId": config.vpc_id,
                    "GroupName": "mario-training",
                }
            ]
        },
    )
    runner.add(
        ["ec2", "describe-key-pairs", "--key-names", config.key_name],
        {"KeyPairs": [{"KeyName": config.key_name}]},
    )
    runner.add(
        [
            "iam",
            "get-instance-profile",
            "--instance-profile-name",
            config.instance_profile,
        ],
        {
            "InstanceProfile": {
                "Path": "/",
                "InstanceProfileName": config.instance_profile,
                "InstanceProfileId": INSTANCE_PROFILE_ID,
                "Arn": _instance_profile_arn(config),
                "CreateDate": "2026-07-24T12:00:00+00:00",
                "Roles": [],
                "Tags": [],
            }
        },
    )
    runner.add(
        [
            "s3api",
            "list-objects-v2",
            "--bucket",
            "defectlens-phase3-002559670021",
            "--prefix",
            "marioai/all32/",
            "--max-items",
            "1",
        ],
        {"KeyCount": 0},
    )
    runner.add(
        [
            "ec2",
            "describe-instances",
            "--filters",
            "Name=tag:Project,Values=MarioAI-All32",
            "Name=instance-state-name,Values=running",
        ],
        {"Reservations": []},
    )
    runner.add(
        _spot_args(config.instance_types),
        {
            "SpotPriceHistory": [
                {
                    "InstanceType": "c7i.8xlarge",
                    "AvailabilityZone": "us-east-1a",
                    "SpotPrice": "0.5568",
                    "Timestamp": NOW.isoformat(),
                }
            ]
        },
    )
    return runner


def _spot_args(instance_types) -> list[str]:
    return [
        "ec2",
        "describe-spot-price-history",
        "--instance-types",
        *sorted(instance_types),
        "--product-descriptions",
        "Linux/UNIX",
        "--start-time",
        (NOW - SPOT_MAX_AGE).isoformat(),
        "--end-time",
        NOW.isoformat(),
    ]


def _preflight_cli(config: AwsConfig) -> tuple[AwsCli, FakeRunner]:
    runner = _successful_runner(config)
    cli = AwsCli(
        profile=config.profile,
        region=config.region,
        runner=runner,
        clock=lambda: NOW,
    )
    return cli, runner


def test_account_configuration_is_exact():
    cfg = AwsConfig.from_yaml(CONFIG_PATH)

    assert cfg.profile == "defectlens"
    assert cfg.account_id == "002559670021"
    assert cfg.region == "us-east-1"
    assert cfg.ami_ssm_parameter == (
        "/aws/service/canonical/ubuntu/server/24.04/stable/current/"
        "amd64/hvm/ebs-gp3/ami-id"
    )
    assert cfg.vpc_id == "vpc-0ce3c6e06be6377df"
    assert cfg.subnet_ids == SUBNET_IDS
    assert cfg.security_group_id == "sg-03fc64395e32dea85"
    assert cfg.key_name == "mario-training-key"
    assert cfg.instance_profile == "defectlens-gpu-role"
    assert cfg.s3_prefix == "s3://defectlens-phase3-002559670021/marioai/all32/"
    assert cfg.instance_types == ("c7i.8xlarge", "c7i.16xlarge")
    assert cfg.on_demand_ceiling_usd == {
        "c7i.8xlarge": Decimal("1.428"),
        "c7i.16xlarge": Decimal("2.856"),
    }
    assert cfg.root_volume_gb == 100
    assert cfg.gp3_monthly_usd_per_gb == Decimal("0.08")
    assert cfg.cap_usd == Decimal("50.00")
    assert cfg.shutdown_threshold_usd == Decimal("49.00")
    assert cfg.grace_minutes == 15
    assert cfg.allocations == {
        "benchmark": Decimal("4.00"),
        "phase_1": Decimal("16.00"),
        "phase_2": Decimal("16.00"),
        "recovery": Decimal("10.00"),
        "evaluation": Decimal("4.00"),
    }


def test_run_uses_fixed_argument_vector_json_and_timeout(config):
    runner = FakeRunner()
    runner.add(["sts", "get-caller-identity"], {"Account": config.account_id})
    cli = AwsCli(profile=config.profile, region=config.region, runner=runner)

    assert cli.run(["sts", "get-caller-identity"]) == {
        "Account": config.account_id
    }
    command, kwargs = runner.calls[0]
    assert command == [
        "aws",
        "sts",
        "get-caller-identity",
        "--profile",
        "defectlens",
        "--region",
        "us-east-1",
        "--output",
        "json",
    ]
    assert kwargs == {
        "check": True,
        "text": True,
        "capture_output": True,
        "timeout": 60,
    }


def test_run_rejects_commands_outside_read_only_allowlist(config):
    cli = AwsCli(profile=config.profile, region=config.region, runner=FakeRunner())

    with pytest.raises(AwsCliError, match="not an allowed read-only operation"):
        cli.run(["ec2", "run-instances", "--image-id", "ami-123"])


def test_run_rejects_equals_form_global_override(config):
    cli = AwsCli(profile=config.profile, region=config.region, runner=FakeRunner())

    with pytest.raises(AwsCliError, match="controlled by AwsCli"):
        cli.run(["sts", "get-caller-identity", "--endpoint-url=https://example.com"])


def test_run_rejects_malformed_json(config):
    runner = FakeRunner()

    def malformed(command, **kwargs):
        return SimpleNamespace(stdout="{bad json", stderr="", returncode=0)

    cli = AwsCli(profile=config.profile, region=config.region, runner=malformed)
    with pytest.raises(AwsCliError, match="invalid JSON"):
        cli.run(["sts", "get-caller-identity"])


def test_run_wraps_timeout_with_actionable_operation(config):
    runner = FakeRunner()
    runner.error = subprocess.TimeoutExpired(
        ["aws", "sts", "get-caller-identity"], timeout=60
    )
    cli = AwsCli(profile=config.profile, region=config.region, runner=runner)

    with pytest.raises(AwsCliError, match=r"sts get-caller-identity.*60 seconds"):
        cli.run(["sts", "get-caller-identity"])


def test_timeout_cannot_be_overridden(config):
    with pytest.raises(TypeError, match="timeout_seconds"):
        AwsCli(
            profile=config.profile,
            region=config.region,
            runner=FakeRunner(),
            timeout_seconds=1,
        )


def test_preflight_validates_every_read_only_prerequisite(config):
    cli, runner = _preflight_cli(config)

    result = cli.preflight(config)

    assert result.account_id == config.account_id
    assert result.vpc_id == config.vpc_id
    assert result.subnet_azs == tuple(
        (subnet_id, f"us-east-1{suffix}")
        for subnet_id, suffix in zip(config.subnet_ids, "abcdef", strict=True)
    )
    assert result.s3_bucket == "defectlens-phase3-002559670021"
    assert result.s3_key_prefix == "marioai/all32/"
    assert result.running_project_instance_ids == ()
    assert result.instance_profile_arn == _instance_profile_arn(config)
    assert result.instance_profile_id == INSTANCE_PROFILE_ID
    assert [call[0][1:3] for call in runner.calls] == [
        ["sts", "get-caller-identity"],
        ["ec2", "describe-vpcs"],
        ["ec2", "describe-subnets"],
        ["ec2", "describe-security-groups"],
        ["ec2", "describe-key-pairs"],
        ["iam", "get-instance-profile"],
        ["s3api", "list-objects-v2"],
        ["ec2", "describe-instances"],
        ["ec2", "describe-spot-price-history"],
    ]


def test_preflight_rejects_when_no_current_allowed_spot_offer(config):
    runner = _successful_runner(config)
    runner.add(_spot_args(config.instance_types), {"SpotPriceHistory": []})
    cli = AwsCli(
        profile=config.profile,
        region=config.region,
        runner=runner,
        clock=lambda: NOW,
    )

    with pytest.raises(AwsPreflightError, match="current allowed Spot offer"):
        cli.preflight(config)


def test_preflight_rejects_wrong_account(config):
    cli, runner = _preflight_cli(config)
    runner.add(["sts", "get-caller-identity"], {"Account": "999999999999"})

    with pytest.raises(AwsPreflightError, match="expected AWS account"):
        cli.preflight(config)


def test_preflight_rejects_missing_or_cross_vpc_subnet(config):
    cli, runner = _preflight_cli(config)
    payload = _subnet_payload(config)
    payload["Subnets"][0]["VpcId"] = "vpc-00000000000000000"
    runner.add(
        ["ec2", "describe-subnets", "--subnet-ids", *config.subnet_ids],
        payload,
    )

    with pytest.raises(AwsPreflightError, match="subnets.*expected VPC"):
        cli.preflight(config)


@pytest.mark.parametrize("availability_zone", ["us-east-1", "us-east-1-invalid"])
def test_preflight_rejects_malformed_region_availability_zone(
    config, availability_zone
):
    cli, runner = _preflight_cli(config)
    payload = _subnet_payload(config)
    payload["Subnets"][0]["AvailabilityZone"] = availability_zone
    runner.add(
        ["ec2", "describe-subnets", "--subnet-ids", *config.subnet_ids],
        payload,
    )

    with pytest.raises(AwsPreflightError, match="invalid availability zone"):
        cli.preflight(config)


def test_preflight_normalizes_unhashable_subnet_id(config):
    cli, runner = _preflight_cli(config)
    payload = _subnet_payload(config)
    payload["Subnets"][0]["SubnetId"] = []
    runner.add(
        ["ec2", "describe-subnets", "--subnet-ids", *config.subnet_ids],
        payload,
    )

    with pytest.raises(AwsPreflightError, match="invalid SubnetId"):
        cli.preflight(config)


def test_preflight_rejects_running_project_instance(config):
    cli, runner = _preflight_cli(config)
    runner.add(
        [
            "ec2",
            "describe-instances",
            "--filters",
            "Name=tag:Project,Values=MarioAI-All32",
            "Name=instance-state-name,Values=running",
        ],
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-0123456789abcdef0",
                            "State": {"Name": "running"},
                        }
                    ]
                }
            ]
        },
    )

    with pytest.raises(
        AwsPreflightError, match="running Project=MarioAI-All32 instance"
    ):
        cli.preflight(config)


def test_preflight_rejects_malformed_s3_list_response(config):
    cli, runner = _preflight_cli(config)
    runner.add(
        [
            "s3api",
            "list-objects-v2",
            "--bucket",
            "defectlens-phase3-002559670021",
            "--prefix",
            "marioai/all32/",
            "--max-items",
            "1",
        ],
        [],
    )

    with pytest.raises(AwsPreflightError, match="S3 prefix list response"):
        cli.preflight(config)


def test_preflight_rejects_cli_profile_or_region_mismatch(config):
    cli = AwsCli(profile="other", region=config.region, runner=FakeRunner())

    with pytest.raises(AwsPreflightError, match="CLI profile"):
        cli.preflight(config)


def test_failed_recheck_invalidates_prior_preflight(config):
    cli, runner = _preflight_cli(config)
    cli.preflight(config)
    runner.add(["sts", "get-caller-identity"], {"Account": "999999999999"})

    with pytest.raises(AwsPreflightError, match="expected AWS account"):
        cli.preflight(config)
    with pytest.raises(AwsPreflightError, match="successful preflight is required"):
        cli.latest_spot_prices(config.instance_types)


def test_spot_selection_is_deterministic_and_restricted(config):
    cli, runner = _preflight_cli(config)
    cli.preflight(config)
    runner.add(
        _spot_args(config.instance_types),
        {
            "SpotPriceHistory": [
                {
                    "InstanceType": "m7i.8xlarge",
                    "AvailabilityZone": "us-east-1a",
                    "SpotPrice": "0.1000",
                    "Timestamp": "2026-07-24T00:00:00+00:00",
                },
                {
                    "InstanceType": "c7i.8xlarge",
                    "AvailabilityZone": "us-east-1z",
                    "SpotPrice": "0.2000",
                    "Timestamp": "2026-07-24T00:00:00+00:00",
                },
                {
                    "InstanceType": "c7i.8xlarge",
                    "AvailabilityZone": "us-east-1f",
                    "SpotPrice": "0.7000",
                    "Timestamp": "2026-07-23T00:00:00+00:00",
                },
                {
                    "InstanceType": "c7i.16xlarge",
                    "AvailabilityZone": "us-east-1c",
                    "SpotPrice": "0.8762",
                    "Timestamp": "2026-07-24T00:00:00+00:00",
                },
                {
                    "InstanceType": "c7i.8xlarge",
                    "AvailabilityZone": "us-east-1f",
                    "SpotPrice": "0.5568",
                    "Timestamp": "2026-07-24T00:00:00+00:00",
                },
            ]
        },
    )

    offers = cli.latest_spot_prices(reversed(config.instance_types))

    assert [(offer.instance_type, offer.availability_zone) for offer in offers] == [
        ("c7i.8xlarge", "us-east-1f"),
        ("c7i.16xlarge", "us-east-1c"),
    ]
    assert offers[0].subnet_id == config.subnet_ids[5]
    assert offers[0].hourly_usd == Decimal("0.5568")


@pytest.mark.parametrize(
    "timestamp",
    [
        (NOW - SPOT_MAX_AGE - timedelta(seconds=1)).isoformat(),
        (NOW + SPOT_MAX_FUTURE_SKEW + timedelta(seconds=1)).isoformat(),
    ],
)
def test_spot_selection_rejects_stale_or_future_allowed_offer(
    config, timestamp
):
    cli, runner = _preflight_cli(config)
    cli.preflight(config)
    runner.add(
        _spot_args(["c7i.8xlarge"]),
        {
            "SpotPriceHistory": [
                {
                    "InstanceType": "c7i.8xlarge",
                    "AvailabilityZone": "us-east-1a",
                    "SpotPrice": "0.5568",
                    "Timestamp": timestamp,
                }
            ]
        },
    )

    with pytest.raises(AwsPreflightError, match="current freshness window"):
        cli.latest_spot_prices(["c7i.8xlarge"])


def test_spot_selection_rejects_type_not_allowed_by_preflight(config):
    cli, _ = _preflight_cli(config)
    cli.preflight(config)

    with pytest.raises(AwsPreflightError, match="instance types outside configuration"):
        cli.latest_spot_prices(["m7i.8xlarge"])


def test_spot_selection_rejects_non_string_type_cleanly(config):
    cli, _ = _preflight_cli(config)
    cli.preflight(config)

    with pytest.raises(AwsPreflightError, match="non-empty strings"):
        cli.latest_spot_prices(["c7i.8xlarge", None])


def test_spot_selection_normalizes_unhashable_availability_zone(config):
    cli, runner = _preflight_cli(config)
    cli.preflight(config)
    runner.add(
        _spot_args(["c7i.8xlarge"]),
        {
            "SpotPriceHistory": [
                {
                    "InstanceType": "c7i.8xlarge",
                    "AvailabilityZone": [],
                    "SpotPrice": "0.5568",
                    "Timestamp": NOW.isoformat(),
                }
            ]
        },
    )

    with pytest.raises(AwsPreflightError, match="malformed Spot offer"):
        cli.latest_spot_prices(["c7i.8xlarge"])


def test_spot_selection_rejects_malformed_allowed_offer(config):
    cli, runner = _preflight_cli(config)
    cli.preflight(config)
    runner.add(
        _spot_args(["c7i.8xlarge"]),
        {
            "SpotPriceHistory": [
                {
                    "InstanceType": "c7i.8xlarge",
                    "AvailabilityZone": "us-east-1a",
                    "SpotPrice": "not-a-price",
                    "Timestamp": "2026-07-24T00:00:00+00:00",
                }
            ]
        },
    )

    with pytest.raises(AwsPreflightError, match="malformed Spot offer"):
        cli.latest_spot_prices(["c7i.8xlarge"])


def test_launch_sets_one_time_spot_and_terminate_on_shutdown(orchestrator):
    instance = orchestrator.launch_guarded_instance(
        phase="benchmark", max_hours=Decimal("1.0")
    )

    request = orchestrator.aws.last_run_instances_request
    assert request["InstanceMarketOptions"]["SpotOptions"]["SpotInstanceType"] == (
        "one-time"
    )
    assert request["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert request["TagSpecifications"][0]["Tags"] == [
        {"Key": "Project", "Value": "MarioAI-All32"},
        {"Key": "Phase", "Value": "benchmark"},
    ]
    assert request["ImageId"] == "ami-0123456789abcdef0"
    assert request["InstanceType"] == "c7i.8xlarge"
    assert request["Placement"] == {"AvailabilityZone": "us-east-1a"}
    assert request["KeyName"] == orchestrator.config.key_name
    assert request["IamInstanceProfile"] == {
        "Name": orchestrator.config.instance_profile
    }
    assert request["NetworkInterfaces"] == [
        {
            "AssociatePublicIpAddress": True,
            "DeleteOnTermination": True,
            "DeviceIndex": 0,
            "Groups": [orchestrator.config.security_group_id],
            "SubnetId": orchestrator.config.subnet_ids[0],
        }
    ]
    assert request["BlockDeviceMappings"] == [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "DeleteOnTermination": True,
                "Encrypted": True,
                "VolumeSize": 100,
                "VolumeType": "gp3",
            },
        }
    ]
    assert request["ClientToken"] == "task-3-idempotency-token"
    assert instance.instance_id == "i-0123456789abcdef0"


def test_launch_checks_budget_before_run_instances(orchestrator):
    orchestrator.ledger = BudgetLedger(
        cap_usd=Decimal("50.00"),
        spent_usd=Decimal("48.00"),
        allocations=orchestrator.config.allocations,
    )

    with pytest.raises(BudgetExceeded):
        orchestrator.launch_guarded_instance("phase_2", Decimal("2"))

    assert not orchestrator.aws.run_instances_called


def test_launch_requires_current_one_shot_preflight_authorization(config):
    clock = FakeMonotonic()
    aws = FakeLifecycleAws(config)
    orchestrator = AwsOrchestrator(
        config=config,
        aws=aws,
        ledger=BudgetLedger(
            cap_usd=config.cap_usd, allocations=config.allocations
        ),
        remote=FakeRemote(clock),
        monotonic=clock,
        sleeper=lambda _seconds: None,
        client_token_factory=lambda: "task-3-idempotency-token",
    )

    with pytest.raises(AwsLifecycleError, match="preflight.*required"):
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))

    orchestrator.preflight()
    clock.now = 301.0
    with pytest.raises(AwsLifecycleError, match="preflight.*expired"):
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))

    clock.now = 0.0
    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    with pytest.raises(AwsLifecycleError, match="preflight.*required"):
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))


def test_launch_rechecks_authorization_immediately_before_mutation(
    orchestrator,
):
    orchestrator.aws.resolve_ami_hook = lambda: setattr(
        orchestrator.test_clock, "now", 301.0
    )

    with pytest.raises(AwsLifecycleError, match="preflight.*expired"):
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))

    assert not orchestrator.aws.run_instances_called


def test_launch_selects_cheapest_offer_and_pins_observed_rate(orchestrator):
    orchestrator.aws.offers = (
        SpotOffer(
            instance_type="c7i.16xlarge",
            availability_zone="us-east-1b",
            subnet_id=orchestrator.config.subnet_ids[1],
            hourly_usd=Decimal("0.8762"),
            timestamp=NOW,
        ),
        SpotOffer(
            instance_type="c7i.8xlarge",
            availability_zone="us-east-1a",
            subnet_id=orchestrator.config.subnet_ids[0],
            hourly_usd=Decimal("0.5568"),
            timestamp=NOW,
        ),
    )

    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )

    assert instance.instance_type == "c7i.8xlarge"
    assert instance.spot_hourly_usd == Decimal("0.5568")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "exactly one instance"),
        ({"Instances": []}, "exactly one instance"),
        ({"Instances": [{}]}, "valid instance ID"),
        ({"Instances": [{"InstanceId": []}]}, "valid instance ID"),
    ],
)
def test_launch_rejects_partial_or_malformed_response(
    orchestrator, response, message
):
    orchestrator.aws.run_instances_response = response

    with pytest.raises(
        AwsLifecycleError,
        match=rf"{message}.*reconcile.*ClientToken",
    ):
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))

    assert (
        orchestrator.aws.last_run_instances_request["ClientToken"]
        == "task-3-idempotency-token"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ClientToken", "different-token"),
        ("ImageId", "ami-11111111111111111"),
        ("InstanceType", "c7i.16xlarge"),
        ("InstanceLifecycle", "scheduled"),
        ("KeyName", "different-key"),
        ("Placement", {"AvailabilityZone": "us-east-1b"}),
        ("SubnetId", "subnet-11111111111111111"),
        ("SecurityGroups", [{"GroupId": "sg-11111111111111111"}]),
        ("Tags", [{"Key": "Project", "Value": "Other"}]),
    ],
)
def test_launch_response_must_correlate_to_exact_request(
    orchestrator, field, value
):
    response = json.loads(
        json.dumps(orchestrator.aws.run_instances_response)
    )
    response["Instances"][0][field] = value
    orchestrator.aws.run_instances_response = response

    with pytest.raises(
        AwsLifecycleError,
        match=r"response.*request.*reconcile.*ClientToken",
    ):
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))


def test_launch_accepts_real_profile_shape_from_immediate_preflight(
    orchestrator,
):
    final_profile_id = "AIPATESTFINALPREFLIGHT"
    orchestrator.aws.instance_profile_id = final_profile_id

    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )

    assert instance.instance_id == "i-0123456789abcdef0"
    assert orchestrator.aws.run_instances_response["Instances"][0][
        "IamInstanceProfile"
    ] == {
        "Arn": _instance_profile_arn(orchestrator.config),
        "Id": final_profile_id,
    }


@pytest.mark.parametrize(
    "profile",
    [
        {
            "Arn": (
                "arn:aws:iam::002559670021:instance-profile/"
                "defectlens-gpu-role"
            ),
            "Id": "AIPAWRONGPROFILEID",
        },
        {
            "Arn": (
                "arn:aws:iam::002559670021:instance-profile/"
                "other-profile"
            ),
            "Id": INSTANCE_PROFILE_ID,
        },
        {
            "Arn": (
                "arn:aws:iam::002559670021:instance-profile/"
                "defectlens-gpu-role"
            )
        },
        {"Id": INSTANCE_PROFILE_ID},
    ],
)
def test_launch_rejects_real_profile_identity_mismatch(
    orchestrator, profile
):
    response = json.loads(
        json.dumps(orchestrator.aws.run_instances_response)
    )
    response["Instances"][0]["IamInstanceProfile"] = profile
    orchestrator.aws.run_instances_response = response

    with pytest.raises(
        AwsLifecycleError,
        match=r"response.*request.*reconcile.*ClientToken",
    ):
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))


def test_launch_rejects_malformed_ami_before_run_instances(orchestrator):
    orchestrator.aws.ami_id = "not-an-ami"

    with pytest.raises(AwsLifecycleError, match="AMI"):
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))

    assert not orchestrator.aws.run_instances_called


def test_accounting_includes_run_instances_response_latency(
    orchestrator, tmp_path
):
    orchestrator.aws.run_instances_hook = lambda: setattr(
        orchestrator.test_clock, "now", 60.0
    )

    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    run = orchestrator.monitor_and_terminate(
        instance, tmp_path / "aws-spend.json"
    )

    assert run.hours == Decimal("60") / Decimal("3600")


def test_monitor_terminates_after_process_exit(orchestrator, tmp_path):
    instance = orchestrator.launch_guarded_instance(
        phase="benchmark", max_hours=Decimal("1.0")
    )
    orchestrator.remote.exit_code = 1
    orchestrator.test_clock.now = 3600.0

    run = orchestrator.monitor_and_terminate(
        instance, tmp_path / "aws-spend.json"
    )

    assert run.hours == Decimal("1")
    assert orchestrator.aws.terminated_ids == [instance.instance_id]


def test_monitor_polls_at_most_once_per_minute_and_persists_each_poll(
    orchestrator, tmp_path, monkeypatch
):
    instance = orchestrator.launch_guarded_instance(
        phase="benchmark", max_hours=Decimal("1.0")
    )
    orchestrator.remote.poll_results = [
        SimpleNamespace(running=True, exit_code=None),
        SimpleNamespace(running=True, exit_code=None),
        SimpleNamespace(running=False, exit_code=0),
    ]
    saved_hours: list[Decimal] = []
    real_save = BudgetLedger.save

    def recording_save(ledger, path):
        saved_hours.append(ledger.runs[-1].hours)
        real_save(ledger, path)

    monkeypatch.setattr(BudgetLedger, "save", recording_save)

    run = orchestrator.monitor_and_terminate(
        instance, tmp_path / "aws-spend.json"
    )

    assert orchestrator.remote.poll_times == [0.0, 60.0, 120.0]
    assert saved_hours[:3] == [
        Decimal("0"),
        Decimal("60") / Decimal("3600"),
        Decimal("120") / Decimal("3600"),
    ]
    assert saved_hours == sorted(saved_hours)
    assert saved_hours[-1] == Decimal("120") / Decimal("3600")
    assert run.hours == Decimal("120") / Decimal("3600")
    assert BudgetLedger.load(tmp_path / "aws-spend.json").runs[-1] == run
    assert orchestrator.aws.terminated_ids == [instance.instance_id]


def test_monitor_requests_remote_shutdown_at_safety_threshold(
    config, tmp_path
):
    config = replace(config, shutdown_threshold_usd=Decimal("47.80"))
    orchestrator = _make_orchestrator(
        config,
        ledger=BudgetLedger(
            cap_usd=config.cap_usd,
            spent_usd=Decimal("47.40"),
            allocations=config.allocations,
        ),
    )
    instance = orchestrator.launch_guarded_instance(
        phase="benchmark", max_hours=Decimal("1.5")
    )
    orchestrator.remote.poll_results = [
        SimpleNamespace(running=True, exit_code=None)
    ]
    orchestrator.test_clock.now = 3600.0

    orchestrator.monitor_and_terminate(
        instance, tmp_path / "aws-spend.json"
    )

    assert orchestrator.remote.shutdown_requests == [instance.instance_id]
    assert orchestrator.remote.shutdown_deadlines == [
        orchestrator.final_deadline(instance) - Decimal("180")
    ]
    assert orchestrator.aws.terminated_ids == [instance.instance_id]


def test_monitor_terminates_when_poll_or_ledger_save_fails(
    orchestrator, tmp_path, monkeypatch
):
    instance = orchestrator.launch_guarded_instance(
        phase="benchmark", max_hours=Decimal("1.0")
    )
    orchestrator.remote.poll_error = ConnectionError("ssh unavailable")

    with pytest.raises(ConnectionError, match="ssh unavailable"):
        orchestrator.monitor_and_terminate(
            instance, tmp_path / "aws-spend.json"
        )

    assert orchestrator.aws.terminated_ids == [instance.instance_id]

    orchestrator.preflight()
    second = orchestrator.launch_guarded_instance(
        phase="benchmark", max_hours=Decimal("1.0")
    )
    orchestrator.remote.poll_error = None
    monkeypatch.setattr(
        BudgetLedger,
        "save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("ledger disk full")
        ),
    )

    with pytest.raises(OSError, match="ledger disk full"):
        orchestrator.monitor_and_terminate(
            second, tmp_path / "aws-spend.json"
        )

    assert orchestrator.aws.terminated_ids[-1] == second.instance_id


def test_monitor_preserves_original_failure_when_termination_also_fails(
    orchestrator, tmp_path
):
    instance = orchestrator.launch_guarded_instance(
        phase="benchmark", max_hours=Decimal("1.0")
    )
    orchestrator.remote.poll_error = ConnectionError("ssh unavailable")
    orchestrator.aws.termination_error = RuntimeError("terminate failed")

    with pytest.raises(ConnectionError, match="ssh unavailable") as raised:
        orchestrator.monitor_and_terminate(
            instance, tmp_path / "aws-spend.json"
        )

    assert any("termination also failed" in note for note in raised.value.__notes__)


def test_state_changing_adapter_is_specific_validated_and_argument_vector(
    config, orchestrator
):
    from scripts.aws_all32 import AwsCommandAdapter

    runner = FakeRunner()
    readonly = FakeLifecycleAws(config)
    adapter = AwsCommandAdapter(
        config=config, readonly=readonly, runner=runner
    )
    adapter.preflight(config)
    launch_request = orchestrator.aws.last_run_instances_request
    if launch_request is None:
        orchestrator.preflight()
        orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
        launch_request = orchestrator.aws.last_run_instances_request
    launch_json = json.dumps(
        launch_request, sort_keys=True, separators=(",", ":")
    )
    runner.add(
        [
            "ssm",
            "get-parameter",
            "--name",
            config.ami_ssm_parameter,
            "--query",
            "Parameter.Value",
        ],
        "ami-0123456789abcdef0",
    )
    runner.add(
        ["ec2", "run-instances", "--cli-input-json", launch_json],
        orchestrator.aws.run_instances_response,
    )
    runner.add(
        [
            "ec2",
            "terminate-instances",
            "--instance-ids",
            "i-0123456789abcdef0",
        ],
        {
            "TerminatingInstances": [
                {
                    "InstanceId": "i-0123456789abcdef0",
                    "CurrentState": {"Name": "shutting-down"},
                }
            ]
        },
    )

    assert adapter.resolve_ami(config.ami_ssm_parameter) == (
        "ami-0123456789abcdef0"
    )
    assert adapter.run_instances(
        launch_request, max_hours=Decimal("1")
    ) == (
        orchestrator.aws.run_instances_response
    )
    adapter.terminate_instance("i-0123456789abcdef0")

    assert [call[0][1:3] for call in runner.calls] == [
        ["ssm", "get-parameter"],
        ["ec2", "run-instances"],
        ["ec2", "terminate-instances"],
    ]
    assert all(
        kwargs
        == {
            "check": True,
            "text": True,
            "capture_output": True,
            "timeout": 60,
        }
        for _, kwargs in runner.calls
    )
    assert not hasattr(adapter, "run")


def test_state_changing_adapter_rejects_invalid_request_before_runner(config):
    from scripts.aws_all32 import AwsCommandAdapter

    runner = FakeRunner()
    adapter = AwsCommandAdapter(
        config=config,
        readonly=AwsCli(
            profile=config.profile,
            region=config.region,
            runner=FakeRunner(),
        ),
        runner=runner,
    )

    with pytest.raises(AwsLifecycleError, match="exact allowlist"):
        adapter.run_instances(
            {"MinCount": 1, "MaxCount": 1},
            max_hours=Decimal("1"),
        )
    with pytest.raises(AwsLifecycleError, match="instance ID"):
        adapter.terminate_instance("i-good; aws ec2 run-instances")

    assert runner.calls == []


def test_adapter_preflight_cannot_reach_mutating_runner(config):
    from scripts.aws_all32 import AwsCommandAdapter

    lifecycle = FakeLifecycleAws(config)

    def forbidden_runner(*_args, **_kwargs):
        raise AssertionError("preflight reached the mutating runner")

    adapter = AwsCommandAdapter(
        config=config, readonly=lifecycle, runner=forbidden_runner
    )

    result = adapter.preflight(config)

    assert result.account_id == config.account_id


def test_adapter_wraps_ambiguous_launch_timeout_without_retry(
    config, orchestrator
):
    from scripts.aws_all32 import AwsCommandAdapter

    runner = FakeRunner()
    adapter = AwsCommandAdapter(
        config=config,
        readonly=FakeLifecycleAws(config),
        runner=runner,
    )
    adapter.preflight(config)
    runner.add(
        [
            "ssm",
            "get-parameter",
            "--name",
            config.ami_ssm_parameter,
            "--query",
            "Parameter.Value",
        ],
        "ami-0123456789abcdef0",
    )
    adapter.resolve_ami(config.ami_ssm_parameter)
    runner.error = subprocess.TimeoutExpired(
        ["aws", "ec2", "run-instances"], timeout=60
    )
    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    request = orchestrator.aws.last_run_instances_request

    with pytest.raises(
        AwsLifecycleError,
        match=r"run-instances.*ambiguous.*ClientToken",
    ):
        adapter.run_instances(request, max_hours=Decimal("1"))

    assert [
        call[0][1:3]
        for call in runner.calls
        if call[0][1:3] == ["ec2", "run-instances"]
    ] == [["ec2", "run-instances"]]


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "instance_type",
        "key",
        "profile",
        "availability_zone",
        "subnet",
        "security_group",
        "volume_size",
        "network_delete",
        "tags",
        "user_data",
    ],
)
def test_mutation_adapter_rejects_every_non_allowlisted_launch_change(
    config, orchestrator, case
):
    from scripts.aws_all32 import AwsCommandAdapter

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    request = json.loads(
        json.dumps(orchestrator.aws.last_run_instances_request)
    )
    runner = FakeRunner()
    runner.add(
        [
            "ssm",
            "get-parameter",
            "--name",
            config.ami_ssm_parameter,
            "--query",
            "Parameter.Value",
        ],
        "ami-0123456789abcdef0",
    )
    adapter = AwsCommandAdapter(
        config=config,
        readonly=FakeLifecycleAws(config),
        runner=runner,
    )
    adapter.preflight(config)
    adapter.resolve_ami(config.ami_ssm_parameter)
    runner.calls.clear()

    if case == "extra":
        request["DryRun"] = False
    elif case == "instance_type":
        request["InstanceType"] = "m7i.8xlarge"
    elif case == "key":
        request["KeyName"] = "other-key"
    elif case == "profile":
        request["IamInstanceProfile"] = {"Name": "other-role"}
    elif case == "availability_zone":
        request["Placement"] = {"AvailabilityZone": "us-east-1z"}
    elif case == "subnet":
        request["NetworkInterfaces"][0]["SubnetId"] = "subnet-other"
    elif case == "security_group":
        request["NetworkInterfaces"][0]["Groups"] = ["sg-other"]
    elif case == "volume_size":
        request["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] = 1000
    elif case == "network_delete":
        request["NetworkInterfaces"][0]["DeleteOnTermination"] = False
    elif case == "tags":
        request["TagSpecifications"][0]["Tags"][0]["Value"] = "Other"
    elif case == "user_data":
        request["UserData"] = base64.b64encode(
            b"#!/bin/sh\ntrue\n"
        ).decode("ascii")

    with pytest.raises(AwsLifecycleError):
        adapter.run_instances(request, max_hours=Decimal("1"))
    assert runner.calls == []


def test_mutation_adapter_rejects_ambiguous_termination_response(
    config, orchestrator
):
    from scripts.aws_all32 import AwsCommandAdapter

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    request = orchestrator.aws.last_run_instances_request
    request_json = json.dumps(
        request, sort_keys=True, separators=(",", ":")
    )
    instance_id = "i-0123456789abcdef0"
    runner = FakeRunner()
    runner.add(
        [
            "ssm",
            "get-parameter",
            "--name",
            config.ami_ssm_parameter,
            "--query",
            "Parameter.Value",
        ],
        "ami-0123456789abcdef0",
    )
    runner.add(
        ["ec2", "run-instances", "--cli-input-json", request_json],
        orchestrator.aws.run_instances_response,
    )
    runner.add(
        ["ec2", "terminate-instances", "--instance-ids", instance_id],
        {},
    )
    adapter = AwsCommandAdapter(
        config=config,
        readonly=FakeLifecycleAws(config),
        runner=runner,
    )
    adapter.preflight(config)
    adapter.resolve_ami(config.ami_ssm_parameter)
    adapter.run_instances(request, max_hours=Decimal("1"))

    with pytest.raises(AwsLifecycleError, match="ambiguous.*reconcile"):
        adapter.terminate_instance(instance_id)


def test_mutation_adapter_requires_exact_budget_authorized_shutdown_delay(
    config, orchestrator
):
    from scripts.aws_all32 import AwsCommandAdapter, _shutdown_user_data

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    request = json.loads(
        json.dumps(orchestrator.aws.last_run_instances_request)
    )
    request["UserData"] = _shutdown_user_data(Decimal("2"))
    runner = FakeRunner()
    runner.add(
        [
            "ssm",
            "get-parameter",
            "--name",
            config.ami_ssm_parameter,
            "--query",
            "Parameter.Value",
        ],
        "ami-0123456789abcdef0",
    )
    adapter = AwsCommandAdapter(
        config=config,
        readonly=FakeLifecycleAws(config),
        runner=runner,
    )
    adapter.preflight(config)
    adapter.resolve_ami(config.ami_ssm_parameter)
    runner.calls.clear()

    with pytest.raises(AwsLifecycleError, match="shutdown.*max_hours"):
        adapter.run_instances(request, max_hours=Decimal("1"))
    assert runner.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["preflight", "--config", str(CONFIG_PATH)],
        [
            "launch",
            "--config",
            str(CONFIG_PATH),
            "--ledger",
            "reports/aws-spend.json",
            "--phase",
            "benchmark",
            "--max-hours",
            "1.0",
        ],
        [
            "resume",
            "--config",
            str(CONFIG_PATH),
            "--ledger",
            "reports/aws-spend.json",
            "--phase",
            "phase_1",
            "--max-hours",
            "1.0",
            "--checkpoint-s3-uri",
            (
                "s3://defectlens-phase3-002559670021/marioai/all32/"
                "models/all32-phase_1/latest.json"
            ),
        ],
        ["status", "--config", str(CONFIG_PATH)],
        [
            "terminate",
            "--config",
            str(CONFIG_PATH),
            "--instance-id",
            "i-0123456789abcdef0",
        ],
        [
            "reconcile",
            "--config",
            str(CONFIG_PATH),
            "--ledger",
            "reports/aws-spend.json",
        ],
    ],
)
def test_cli_exposes_each_guarded_lifecycle_subcommand(argv):
    from scripts.aws_all32 import build_parser

    args = build_parser().parse_args(argv)

    assert args.command == argv[0]


def test_cli_preflight_dispatches_through_read_only_boundary(config):
    from scripts.aws_all32 import main

    runner = _successful_runner(config)
    stdout = io.StringIO()

    result = main(
        ["preflight", "--config", str(CONFIG_PATH)],
        runner=runner,
        stdout=stdout,
        clock=lambda: NOW,
    )

    assert result == 0
    payload = json.loads(stdout.getvalue())
    assert payload["account_id"] == config.account_id
    assert [call[0][1:3] for call in runner.calls] == [
        ["sts", "get-caller-identity"],
        ["ec2", "describe-vpcs"],
        ["ec2", "describe-subnets"],
        ["ec2", "describe-security-groups"],
        ["ec2", "describe-key-pairs"],
        ["iam", "get-instance-profile"],
        ["s3api", "list-objects-v2"],
        ["ec2", "describe-instances"],
        ["ec2", "describe-spot-price-history"],
    ]


def test_cli_status_uses_read_only_project_query(config):
    from scripts.aws_all32 import main

    runner = FakeRunner()
    runner.add(
        ["sts", "get-caller-identity"],
        {"Account": config.account_id},
    )
    runner.add(
        [
            "ec2",
            "describe-instances",
            "--filters",
            "Name=tag:Project,Values=MarioAI-All32",
        ],
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-0123456789abcdef0",
                            "InstanceType": "c7i.8xlarge",
                            "State": {"Name": "running"},
                            "PublicIpAddress": "203.0.113.10",
                        }
                    ]
                }
            ]
        },
    )
    stdout = io.StringIO()

    result = main(
        ["status", "--config", str(CONFIG_PATH)],
        runner=runner,
        stdout=stdout,
    )

    assert result == 0
    assert json.loads(stdout.getvalue()) == {
        "instances": [
            {
                "instance_id": "i-0123456789abcdef0",
                "instance_type": "c7i.8xlarge",
                "public_ip": "203.0.113.10",
                "state": "running",
            }
        ]
    }
    assert [call[0][1:3] for call in runner.calls] == [
        ["sts", "get-caller-identity"],
        ["ec2", "describe-instances"],
    ]


def test_cli_terminate_and_reconcile_are_idempotently_scoped(config, tmp_path):
    from scripts.aws_all32 import main

    instance_id = "i-0123456789abcdef0"
    terminate_args = [
        "ec2",
        "terminate-instances",
        "--instance-ids",
        instance_id,
    ]
    terminating_payload = {
        "TerminatingInstances": [
            {
                "InstanceId": instance_id,
                "CurrentState": {"Name": "shutting-down"},
            }
        ]
    }
    terminate_runner = FakeRunner()
    terminate_runner.add(
        ["sts", "get-caller-identity"],
        {"Account": config.account_id},
    )
    terminate_runner.add(
        [
            "ec2",
            "describe-instances",
            "--instance-ids",
            instance_id,
            "--filters",
            "Name=tag:Project,Values=MarioAI-All32",
        ],
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": instance_id,
                            "InstanceType": "c7i.8xlarge",
                            "State": {"Name": "running"},
                        }
                    ]
                }
            ]
        },
    )
    terminate_runner.add(terminate_args, terminating_payload)

    assert (
        main(
            [
                "terminate",
                "--config",
                str(CONFIG_PATH),
                "--instance-id",
                instance_id,
            ],
            runner=terminate_runner,
            stdout=io.StringIO(),
        )
        == 0
    )

    reconcile_runner = FakeRunner()
    reconcile_runner.add(
        ["sts", "get-caller-identity"],
        {"Account": config.account_id},
    )
    reconcile_runner.add(
        [
            "ec2",
            "describe-instances",
            "--filters",
            "Name=tag:Project,Values=MarioAI-All32",
                (
                    "Name=instance-state-name,"
                    "Values=pending,running,stopping,stopped,shutting-down"
                ),
        ],
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": instance_id,
                            "InstanceType": "c7i.8xlarge",
                            "State": {"Name": "running"},
                        }
                    ]
                }
            ]
        },
    )
    reconcile_runner.add(terminate_args, terminating_payload)
    ledger_path = tmp_path / "aws-spend.json"
    stdout = io.StringIO()

    assert (
        main(
            [
                "reconcile",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
            ],
            runner=reconcile_runner,
            stdout=stdout,
        )
        == 0
    )

    assert json.loads(stdout.getvalue()) == {
        "cap_usd": "50.00",
        "remaining_usd": "50.00",
        "reservation_cleared": True,
        "spent_usd": "0",
        "terminated_instance_ids": [instance_id],
    }
    assert [call[0][1:3] for call in reconcile_runner.calls] == [
        ["sts", "get-caller-identity"],
        ["ec2", "describe-instances"],
        ["ec2", "terminate-instances"],
    ]


def test_cli_terminate_does_not_mutate_an_unowned_or_missing_instance(config):
    from scripts.aws_all32 import main

    instance_id = "i-0123456789abcdef0"
    runner = FakeRunner()
    runner.add(
        ["sts", "get-caller-identity"],
        {"Account": config.account_id},
    )
    runner.add(
        [
            "ec2",
            "describe-instances",
            "--instance-ids",
            instance_id,
            "--filters",
            "Name=tag:Project,Values=MarioAI-All32",
        ],
        {"Reservations": []},
    )
    stdout = io.StringIO()

    assert (
        main(
            [
                "terminate",
                "--config",
                str(CONFIG_PATH),
                "--instance-id",
                instance_id,
            ],
            runner=runner,
            stdout=stdout,
        )
        == 0
    )

    assert json.loads(stdout.getvalue()) == {
        "instance_id": instance_id,
        "termination_requested": False,
    }
    assert [call[0][1:3] for call in runner.calls] == [
        ["sts", "get-caller-identity"],
        ["ec2", "describe-instances"],
    ]


def test_cli_launch_preflights_starts_monitors_and_terminates(config, tmp_path):
    from scripts.aws_all32 import main

    clock = FakeMonotonic()
    aws = FakeLifecycleAws(config)
    remote = FakeRemote(clock)
    stdout = io.StringIO()
    ledger_path = tmp_path / "aws-spend.json"

    result = main(
        [
            "launch",
            "--config",
            str(CONFIG_PATH),
            "--ledger",
            str(ledger_path),
            "--phase",
            "benchmark",
            "--max-hours",
            "1.0",
        ],
        stdout=stdout,
        aws_override=aws,
        remote=remote,
        monotonic=clock,
        sleeper=lambda _seconds: None,
        client_token_factory=lambda: "task-3-idempotency-token",
    )

    assert result == 0
    assert aws.run_instances_called
    assert remote.started == [
        (
            "i-0123456789abcdef0",
            "benchmark",
            3600,
            config.s3_prefix,
        )
    ]
    assert aws.terminated_ids == ["i-0123456789abcdef0"]
    assert BudgetLedger.load(ledger_path).runs[0].instance_id == (
        "i-0123456789abcdef0"
    )
    payload = json.loads(stdout.getvalue())
    assert payload["instance_id"] == "i-0123456789abcdef0"
    assert payload["costed_run"]["instance_hourly_usd"] == "1.428"


def test_cli_launch_terminates_when_remote_start_fails(config, tmp_path):
    from scripts.aws_all32 import main

    clock = FakeMonotonic()
    aws = FakeLifecycleAws(config)
    remote = FakeRemote(clock)
    remote.start_error = ConnectionError("rsync failed")

    with pytest.raises(ConnectionError, match="rsync failed"):
        main(
            [
                "launch",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(tmp_path / "aws-spend.json"),
                "--phase",
                "benchmark",
                "--max-hours",
                "1.0",
            ],
            stdout=io.StringIO(),
            aws_override=aws,
            remote=remote,
            monotonic=clock,
            sleeper=lambda _seconds: None,
            client_token_factory=lambda: "task-3-idempotency-token",
        )

    assert aws.terminated_ids == ["i-0123456789abcdef0"]


def test_cli_launch_persists_exact_reservation_before_mutation(
    config, tmp_path
):
    from scripts.aws_all32 import main

    clock = FakeMonotonic()
    aws = FakeLifecycleAws(config)
    remote = FakeRemote(clock)
    ledger_path = tmp_path / "aws-spend.json"
    state_path = ledger_path.with_name(f"{ledger_path.name}.launch.json")
    observed = {}

    def assert_reserved_before_mutation():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        observed.update(payload)
        assert payload["state"] == "reserved"
        assert payload["instance_id"] is None
        assert payload["request"]["ClientToken"] == payload["client_token"]

    aws.run_instances_hook = assert_reserved_before_mutation

    assert (
        main(
            [
                "launch",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
                "--phase",
                "benchmark",
                "--max-hours",
                "1",
            ],
            stdout=io.StringIO(),
            aws_override=aws,
            remote=remote,
            monotonic=clock,
            sleeper=lambda _seconds: None,
            wall_clock=lambda: 1234.5,
            client_token_factory=lambda: "durable-stable-token",
        )
        == 0
    )

    assert observed["requested_epoch_seconds"] == "1234.5"
    assert observed["on_demand_hourly_usd"] == "1.428"
    assert "shutdown -h +" in base64.b64decode(
        observed["request"]["UserData"], validate=True
    ).decode("utf-8")
    assert not state_path.exists()


def test_reservation_write_failure_prevents_run_instances(
    config, tmp_path, monkeypatch
):
    from scripts.aws_all32 import LaunchStateStore, main

    aws = FakeLifecycleAws(config)

    def fail_save(_self, _reservation):
        raise OSError("disk full")

    monkeypatch.setattr(LaunchStateStore, "save", fail_save)

    with pytest.raises(OSError, match="disk full"):
        main(
            [
                "launch",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(tmp_path / "aws-spend.json"),
                "--phase",
                "benchmark",
                "--max-hours",
                "1",
            ],
            stdout=io.StringIO(),
            aws_override=aws,
            remote=FakeRemote(),
            client_token_factory=lambda: "never-mutated-token",
        )

    assert not aws.run_instances_called


def test_post_launch_state_save_failure_still_accounts_and_terminates(
    config, tmp_path, monkeypatch
):
    from scripts.aws_all32 import LaunchStateStore, main

    aws = FakeLifecycleAws(config)
    real_save = LaunchStateStore.save

    def fail_launched_save(store, reservation):
        if reservation.state == "launched":
            raise OSError("launched state fsync failed")
        real_save(store, reservation)

    monkeypatch.setattr(LaunchStateStore, "save", fail_launched_save)
    ledger_path = tmp_path / "aws-spend.json"

    with pytest.raises(OSError, match="launched state fsync failed"):
        main(
            [
                "launch",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
                "--phase",
                "benchmark",
                "--max-hours",
                "1",
            ],
            stdout=io.StringIO(),
            aws_override=aws,
            remote=FakeRemote(),
            monotonic=FakeMonotonic(),
            client_token_factory=lambda: "post-launch-save-token",
        )

    assert aws.terminated_ids == ["i-0123456789abcdef0"]
    ledger = BudgetLedger.load(ledger_path, cap_usd=config.cap_usd)
    assert ledger.runs[0].instance_id == "i-0123456789abcdef0"
    assert ledger.runs[0].instance_hourly_usd == Decimal("1.428")


def test_launch_runs_fresh_full_preflight_immediately_before_mutation(
    orchestrator,
):
    calls_at_mutation = []
    orchestrator.aws.run_instances_hook = lambda: calls_at_mutation.append(
        orchestrator.aws.preflight_calls
    )

    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))

    assert calls_at_mutation == [2]


def test_ambiguous_launch_keeps_stable_reservation_and_blocks_retry(
    config, tmp_path
):
    from scripts.aws_all32 import main

    ledger_path = tmp_path / "aws-spend.json"
    state_path = ledger_path.with_name(f"{ledger_path.name}.launch.json")
    first_aws = FakeLifecycleAws(config)
    first_aws.run_instances_error = TimeoutError("ambiguous launch")

    with pytest.raises(TimeoutError, match="ambiguous launch"):
        main(
            [
                "launch",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
                "--phase",
                "benchmark",
                "--max-hours",
                "1",
            ],
            stdout=io.StringIO(),
            aws_override=first_aws,
            remote=FakeRemote(),
            client_token_factory=lambda: "first-stable-token",
        )

    reservation = json.loads(state_path.read_text(encoding="utf-8"))
    assert reservation["client_token"] == "first-stable-token"
    second_aws = FakeLifecycleAws(config)
    with pytest.raises(AwsLifecycleError, match="reservation.*reconcile"):
        main(
            [
                "launch",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
                "--phase",
                "benchmark",
                "--max-hours",
                "1",
            ],
            stdout=io.StringIO(),
            aws_override=second_aws,
            remote=FakeRemote(),
            client_token_factory=lambda: "must-not-be-used",
        )
    assert not second_aws.run_instances_called


def test_launch_lock_serializes_callers(config, tmp_path):
    from scripts.aws_all32 import LaunchStateStore, main

    ledger_path = tmp_path / "aws-spend.json"
    aws = FakeLifecycleAws(config)
    with LaunchStateStore(ledger_path):
        with pytest.raises(AwsLifecycleError, match="another AWS lifecycle"):
            main(
                [
                    "launch",
                    "--config",
                    str(CONFIG_PATH),
                    "--ledger",
                    str(ledger_path),
                    "--phase",
                    "benchmark",
                    "--max-hours",
                    "1",
                ],
                stdout=io.StringIO(),
                aws_override=aws,
                remote=FakeRemote(),
            )
    assert not aws.run_instances_called


def test_pending_project_instance_blocks_before_reservation(config, tmp_path):
    from scripts.aws_all32 import main

    aws = FakeLifecycleAws(config)
    aws.project_results = [
        (
            {
                "instance_id": "i-0123456789abcdef0",
                "instance_type": "c7i.8xlarge",
                "public_ip": None,
                "state": "pending",
            },
        )
    ]
    ledger_path = tmp_path / "aws-spend.json"

    with pytest.raises(AwsLifecycleError, match="active.*reconcile"):
        main(
            [
                "launch",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
                "--phase",
                "benchmark",
                "--max-hours",
                "1",
            ],
            stdout=io.StringIO(),
            aws_override=aws,
            remote=FakeRemote(),
        )

    assert not aws.run_instances_called
    assert not ledger_path.with_name(
        f"{ledger_path.name}.launch.json"
    ).exists()


def test_reconcile_settles_reservation_idempotently_before_clearing(
    config, orchestrator, tmp_path
):
    from scripts.aws_all32 import LaunchReservation, LaunchStateStore, main

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    request = orchestrator.aws.last_run_instances_request
    ledger_path = tmp_path / "aws-spend.json"
    reservation = LaunchReservation(
        client_token="reconcile-stable-token",
        state="launched",
        phase="benchmark",
        request=request,
        instance_hourly_usd=Decimal("0.5568"),
        on_demand_hourly_usd=Decimal("1.428"),
        volume_hourly_usd=Decimal("8") / Decimal("720"),
        max_hours=Decimal("1"),
        grace_hours=Decimal("0.25"),
        requested_epoch_seconds=Decimal("1000"),
        instance_id="i-0123456789abcdef0",
    )
    with LaunchStateStore(ledger_path) as store:
        store.save(reservation)
    aws = FakeLifecycleAws(config)
    instance_summary = {
        "instance_id": "i-0123456789abcdef0",
        "instance_type": "c7i.8xlarge",
        "public_ip": "203.0.113.10",
        "state": "running",
    }
    aws.project_results = [
        (instance_summary,),
        (instance_summary,),
        (),
    ]

    assert (
        main(
            [
                "reconcile",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
            ],
            stdout=io.StringIO(),
            aws_override=aws,
        )
        == 0
    )
    first = BudgetLedger.load(ledger_path, cap_usd=config.cap_usd)
    expected = Decimal("1.25") * (
        Decimal("1.428") + Decimal("8") / Decimal("720")
    )
    assert first.spent_usd == expected
    assert {run.phase for run in first.runs} == {
        "benchmark",
        "__launch_grace__",
    }
    assert not store.state_path.exists()

    assert (
        main(
            [
                "reconcile",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
            ],
            stdout=io.StringIO(),
            aws_override=FakeLifecycleAws(config),
        )
        == 0
    )
    assert BudgetLedger.load(
        ledger_path, cap_usd=config.cap_usd
    ).spent_usd == expected


def test_reconcile_keeps_ambiguous_unknown_reservation_fail_closed(
    config, orchestrator, tmp_path
):
    from scripts.aws_all32 import LaunchReservation, LaunchStateStore, main

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    ledger_path = tmp_path / "aws-spend.json"
    reservation = LaunchReservation(
        client_token="unknown-stable-token",
        state="reserved",
        phase="benchmark",
        request=orchestrator.aws.last_run_instances_request,
        instance_hourly_usd=Decimal("0.5568"),
        on_demand_hourly_usd=Decimal("1.428"),
        volume_hourly_usd=Decimal("8") / Decimal("720"),
        max_hours=Decimal("1"),
        grace_hours=Decimal("0.25"),
        requested_epoch_seconds=Decimal("1000"),
    )
    with LaunchStateStore(ledger_path) as store:
        store.save(reservation)

    stdout = io.StringIO()
    assert (
        main(
            [
                "reconcile",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
            ],
            stdout=stdout,
            aws_override=FakeLifecycleAws(config),
        )
        == 0
    )

    assert json.loads(stdout.getvalue())["reservation_cleared"] is False
    assert store.state_path.exists()
    spent = BudgetLedger.load(
        ledger_path, cap_usd=config.cap_usd
    ).spent_usd
    assert spent > 0

    with pytest.raises(AwsLifecycleError, match="reservation.*reconcile"):
        main(
            [
                "launch",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
                "--phase",
                "benchmark",
                "--max-hours",
                "1",
            ],
            stdout=io.StringIO(),
            aws_override=FakeLifecycleAws(config),
            remote=FakeRemote(),
        )


def test_reconcile_never_substitutes_unrelated_project_instance(
    config, orchestrator, tmp_path
):
    from scripts.aws_all32 import LaunchReservation, LaunchStateStore, main

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    ledger_path = tmp_path / "aws-spend.json"
    reservation = LaunchReservation(
        client_token="unknown-owned-token",
        state="reserved",
        phase="benchmark",
        request=orchestrator.aws.last_run_instances_request,
        instance_hourly_usd=Decimal("0.5568"),
        on_demand_hourly_usd=Decimal("1.428"),
        volume_hourly_usd=Decimal("8") / Decimal("720"),
        max_hours=Decimal("1"),
        grace_hours=Decimal("0.25"),
        requested_epoch_seconds=Decimal("1000"),
    )
    with LaunchStateStore(ledger_path) as store:
        store.save(reservation)
    unrelated = {
        "instance_id": "i-11111111111111111",
        "instance_type": "c7i.8xlarge",
        "public_ip": "203.0.113.111",
        "state": "running",
    }
    aws = FakeLifecycleAws(config)
    aws.project_results = [(unrelated,), (), ()]

    assert (
        main(
            [
                "reconcile",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
            ],
            stdout=io.StringIO(),
            aws_override=aws,
        )
        == 0
    )

    assert store.state_path.exists()
    ledger = BudgetLedger.load(ledger_path, cap_usd=config.cap_usd)
    main_runs = [run for run in ledger.runs if run.phase == "benchmark"]
    assert [run.instance_id for run in main_runs] == [
        "pending:unknown-owned-token"
    ]
    assert aws.terminated_ids == ["i-11111111111111111"]


def test_pending_settlement_migrates_to_recovered_instance_without_reuse(
    config, orchestrator
):
    from scripts.aws_all32 import LaunchReservation, _settle_reservation

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    reservation = LaunchReservation(
        client_token="migration-token",
        state="reserved",
        phase="benchmark",
        request=orchestrator.aws.last_run_instances_request,
        instance_hourly_usd=Decimal("0.5568"),
        on_demand_hourly_usd=Decimal("1.428"),
        volume_hourly_usd=Decimal("8") / Decimal("720"),
        max_hours=Decimal("1"),
        grace_hours=Decimal("0.25"),
        requested_epoch_seconds=Decimal("1000"),
    )
    empty = BudgetLedger(
        cap_usd=config.cap_usd, allocations=config.allocations
    )
    pending = _settle_reservation(
        empty, reservation, instance_id=None
    )

    migrated = _settle_reservation(
        pending,
        reservation,
        instance_id="i-0123456789abcdef0",
    )

    assert migrated.spent_usd == pending.spent_usd
    assert [
        run.instance_id
        for run in migrated.runs
        if run.phase == "benchmark"
    ] == ["i-0123456789abcdef0"]


def test_migrated_settlement_stays_real_when_token_later_disappears(
    config, orchestrator
):
    from scripts.aws_all32 import LaunchReservation, _settle_reservation

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    instance_id = "i-0123456789abcdef0"
    reservation = LaunchReservation(
        client_token="migration-retry-token",
        state="reserved",
        phase="benchmark",
        request=orchestrator.aws.last_run_instances_request,
        instance_hourly_usd=Decimal("0.5568"),
        on_demand_hourly_usd=Decimal("1.428"),
        volume_hourly_usd=Decimal("8") / Decimal("720"),
        max_hours=Decimal("1"),
        grace_hours=Decimal("0.25"),
        requested_epoch_seconds=Decimal("1000"),
    )
    empty = BudgetLedger(
        cap_usd=config.cap_usd, allocations=config.allocations
    )
    pending = _settle_reservation(empty, reservation, instance_id=None)

    retried = _settle_reservation(
        pending,
        replace(reservation, state="launched", instance_id=instance_id),
        instance_id=None,
    )

    assert retried.spent_usd == pending.spent_usd
    assert [
        run.instance_id
        for run in retried.runs
        if run.phase == "benchmark"
    ] == [instance_id]


def test_reconcile_persists_recovered_identity_before_termination(
    config, orchestrator, tmp_path
):
    from scripts.aws_all32 import LaunchReservation, LaunchStateStore, main

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    ledger_path = tmp_path / "aws-spend.json"
    instance_id = "i-0123456789abcdef0"
    reservation = LaunchReservation(
        client_token="recovered-before-mutation-token",
        state="reserved",
        phase="benchmark",
        request=orchestrator.aws.last_run_instances_request,
        instance_hourly_usd=Decimal("0.5568"),
        on_demand_hourly_usd=Decimal("1.428"),
        volume_hourly_usd=Decimal("8") / Decimal("720"),
        max_hours=Decimal("1"),
        grace_hours=Decimal("0.25"),
        requested_epoch_seconds=Decimal("1000"),
    )
    with LaunchStateStore(ledger_path) as store:
        store.save(reservation)
    recovered = {
        "instance_id": instance_id,
        "instance_type": "c7i.8xlarge",
        "public_ip": "203.0.113.10",
        "state": "running",
    }
    terminated = {**recovered, "state": "terminated"}
    aws = FakeLifecycleAws(config)
    aws.project_results = [(), (recovered,), (terminated,)]
    state_seen_at_termination = []

    def record_durable_identity():
        state_seen_at_termination.append(
            json.loads(store.state_path.read_text(encoding="utf-8"))
        )

    aws.terminate_hook = record_durable_identity

    assert (
        main(
            [
                "reconcile",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
            ],
            stdout=io.StringIO(),
            aws_override=aws,
        )
        == 0
    )

    assert state_seen_at_termination == [
        {
            **reservation.to_dict(),
            "instance_id": instance_id,
            "state": "launched",
        }
    ]


def test_reconcile_requires_terminal_confirmation_before_sidecar_clear(
    config, orchestrator, tmp_path
):
    from scripts.aws_all32 import LaunchReservation, LaunchStateStore, main

    orchestrator.preflight()
    orchestrator.launch_guarded_instance("benchmark", Decimal("1"))
    ledger_path = tmp_path / "aws-spend.json"
    instance_id = "i-0123456789abcdef0"
    reservation = LaunchReservation(
        client_token="terminal-confirmation-token",
        state="launched",
        phase="benchmark",
        request=orchestrator.aws.last_run_instances_request,
        instance_hourly_usd=Decimal("0.5568"),
        on_demand_hourly_usd=Decimal("1.428"),
        volume_hourly_usd=Decimal("8") / Decimal("720"),
        max_hours=Decimal("1"),
        grace_hours=Decimal("0.25"),
        requested_epoch_seconds=Decimal("1000"),
        instance_id=instance_id,
    )
    with LaunchStateStore(ledger_path) as store:
        store.save(reservation)
    running = {
        "instance_id": instance_id,
        "instance_type": "c7i.8xlarge",
        "public_ip": "203.0.113.10",
        "state": "running",
    }
    aws = FakeLifecycleAws(config)
    aws.project_results = [(running,), (running,), (), (running,)]

    assert (
        main(
            [
                "reconcile",
                "--config",
                str(CONFIG_PATH),
                "--ledger",
                str(ledger_path),
            ],
            stdout=io.StringIO(),
            aws_override=aws,
        )
        == 0
    )

    assert store.state_path.exists()


def test_wait_for_running_public_ip_retries_eventual_consistency(
    orchestrator,
):
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    orchestrator.aws.project_results = [
        (),
        (
            {
                "instance_id": instance.instance_id,
                "instance_type": instance.instance_type,
                "public_ip": None,
                "state": "pending",
            },
        ),
        (
            {
                "instance_id": instance.instance_id,
                "instance_type": instance.instance_type,
                "public_ip": "203.0.113.99",
                "state": "running",
            },
        ),
    ]

    ready = orchestrator.wait_for_running_public_ip(instance)

    assert ready.public_ip == "203.0.113.99"
    assert orchestrator.test_clock.now == 10.0


def test_final_accounting_includes_termination_latency(orchestrator, tmp_path):
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    orchestrator.aws.terminate_hook = lambda: setattr(
        orchestrator.test_clock, "now", 120.0
    )

    run = orchestrator.monitor_and_terminate(
        instance, tmp_path / "aws-spend.json"
    )

    assert run.hours == Decimal("120") / Decimal("3600")
    assert BudgetLedger.load(tmp_path / "aws-spend.json").runs[0].hours == (
        run.hours
    )


def test_successful_run_accounts_at_conservative_on_demand_ceiling(
    orchestrator, tmp_path
):
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    orchestrator.test_clock.now = 60.0

    run = orchestrator.monitor_and_terminate(
        instance, tmp_path / "aws-spend.json"
    )

    assert instance.spot_hourly_usd == Decimal("0.5568")
    assert run.instance_hourly_usd == Decimal("1.428")
    assert BudgetLedger.load(
        tmp_path / "aws-spend.json"
    ).runs[0].instance_hourly_usd == Decimal("1.428")


def test_ec2_settlement_uses_only_remaining_absolute_grace(
    orchestrator, tmp_path
):
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    orchestrator.test_clock.now = 4440.0
    shutting_down = {
        "instance_id": instance.instance_id,
        "instance_type": instance.instance_type,
        "public_ip": instance.public_ip,
        "state": "shutting-down",
    }
    orchestrator.aws.project_results = [(shutting_down,)] * 100

    with pytest.raises(AwsLifecycleError, match="absolute.*deadline"):
        orchestrator.terminate_and_settle(
            instance, tmp_path / "aws-spend.json"
        )

    assert orchestrator.test_clock.now == 4500.0


def test_ec2_settlement_bounds_each_read_and_never_starts_after_deadline(
    orchestrator, tmp_path
):
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    orchestrator.test_clock.now = 4499.5
    shutting_down = {
        "instance_id": instance.instance_id,
        "instance_type": instance.instance_type,
        "public_ip": instance.public_ip,
        "state": "shutting-down",
    }
    read_timeouts = []
    sleeps = []

    def bounded_read(
        *,
        instance_id=None,
        client_token=None,
        active_only=False,
        timeout_seconds=None,
    ):
        del client_token, active_only
        assert instance_id == instance.instance_id
        read_timeouts.append(timeout_seconds)
        return (shutting_down,)

    def bounded_sleep(seconds):
        sleeps.append(seconds)
        orchestrator.test_clock.now += seconds

    orchestrator.aws.project_instances = bounded_read
    orchestrator._sleeper = bounded_sleep

    with pytest.raises(AwsLifecycleError, match="absolute.*deadline"):
        orchestrator.terminate_and_settle(
            instance, tmp_path / "aws-spend.json"
        )

    assert read_timeouts == [pytest.approx(0.5)]
    assert sleeps == [pytest.approx(0.5)]
    assert orchestrator.test_clock.now == 4500.0


def test_project_instance_read_propagates_bounded_timeout(
    config, orchestrator
):
    from scripts.aws_all32 import AwsCommandAdapter

    runner = FakeRunner()
    runner.add(
        [
            "ec2",
            "describe-instances",
            "--instance-ids",
            "i-0123456789abcdef0",
            "--filters",
            "Name=tag:Project,Values=MarioAI-All32",
        ],
        {"Reservations": []},
    )
    adapter = AwsCommandAdapter(
        config=config,
        readonly=AwsCli(
            profile=config.profile,
            region=config.region,
            runner=runner,
        ),
        runner=FakeRunner(),
    )

    assert adapter.project_instances(
        instance_id="i-0123456789abcdef0",
        timeout_seconds=0.25,
    ) == ()
    assert runner.calls[0][1]["timeout"] == pytest.approx(0.25)


def test_readiness_persists_paid_cost_during_waits(
    orchestrator, tmp_path, monkeypatch
):
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    pending = {
        "instance_id": instance.instance_id,
        "instance_type": instance.instance_type,
        "public_ip": None,
        "state": "pending",
    }
    ready = {
        **pending,
        "public_ip": "203.0.113.77",
        "state": "running",
    }
    orchestrator.aws.project_results = [
        (pending,),
        (pending,),
        (pending,),
        (ready,),
    ]
    save_times = []
    real_save = BudgetLedger.save

    def recording_save(ledger, path):
        save_times.append(orchestrator.test_clock.now)
        real_save(ledger, path)

    monkeypatch.setattr(BudgetLedger, "save", recording_save)

    result = orchestrator.wait_for_running_public_ip(
        instance, ledger_path=tmp_path / "aws-spend.json"
    )

    assert result.public_ip == "203.0.113.77"
    assert save_times == [0.0, 5.0, 10.0, 15.0]
    assert max(
        later - earlier
        for earlier, later in zip(save_times, save_times[1:])
    ) <= 60


def test_paid_subprocess_ticks_at_most_every_sixty_seconds(
    orchestrator, tmp_path
):
    from scripts.aws_all32 import SshRemoteSupervisor

    ssh_key = tmp_path / "mario-training-key.pem"
    ssh_key.write_text("test-only", encoding="utf-8")
    local_repo = tmp_path / "MarioAI"
    local_repo.mkdir()
    clock = FakeMonotonic()
    communicate_timeouts = []
    ticks = []

    class SlowProcess:
        returncode = 0

        def __init__(self):
            self.calls = 0

        def communicate(self, input=None, timeout=None):
            del input
            communicate_timeouts.append(timeout)
            self.calls += 1
            if self.calls <= 2:
                clock.now += timeout
                raise subprocess.TimeoutExpired(["ssh"], timeout)
            return ("complete", "")

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    remote = SshRemoteSupervisor(
        aws=orchestrator.aws,
        ssh_key=ssh_key,
        local_repo=local_repo,
        monotonic=clock,
        sleeper=lambda _seconds: None,
        process_factory=lambda *_args, **_kwargs: SlowProcess(),
    )

    remote._run_remote_command(
        ["ssh", "example"],
        timeout=300,
        input_text="test\n",
        on_tick=lambda: ticks.append(clock.now),
        absolute_deadline=Decimal("180"),
    )

    assert communicate_timeouts == [60, 60, 60]
    assert ticks == [60.0, 120.0, 120.0]


def test_paid_subprocess_reaps_live_child_when_tick_raises(
    orchestrator, tmp_path
):
    from scripts.aws_all32 import SshRemoteSupervisor

    ssh_key = tmp_path / "mario-training-key.pem"
    ssh_key.write_text("test-only", encoding="utf-8")
    local_repo = tmp_path / "MarioAI"
    local_repo.mkdir()
    clock = FakeMonotonic()

    class LiveProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.waited = False

        def communicate(self, input=None, timeout=None):
            del input
            clock.now += timeout
            raise subprocess.TimeoutExpired(["rsync"], timeout)

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            self.waited = True
            return self.returncode

    process = LiveProcess()
    remote = SshRemoteSupervisor(
        aws=orchestrator.aws,
        ssh_key=ssh_key,
        local_repo=local_repo,
        monotonic=clock,
        sleeper=lambda _seconds: None,
        process_factory=lambda *_args, **_kwargs: process,
    )

    with pytest.raises(RuntimeError, match="accounting failed"):
        remote._run_remote_command(
            ["rsync", "source", "target"],
            timeout=300,
            on_tick=lambda: (_ for _ in ()).throw(
                RuntimeError("accounting failed")
            ),
            absolute_deadline=Decimal("180"),
        )

    assert process.terminated is True
    assert process.waited is True


def test_ssh_remote_start_uses_argument_vectors(orchestrator, tmp_path):
    from scripts.aws_all32 import SshRemoteSupervisor

    orchestrator.preflight()
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    ssh_key = tmp_path / "mario-training-key.pem"
    ssh_key.write_text("test-only", encoding="utf-8")
    local_repo = tmp_path / "MarioAI"
    local_repo.mkdir()
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    remote = SshRemoteSupervisor(
        aws=orchestrator.aws,
        ssh_key=ssh_key,
        local_repo=local_repo,
        runner=runner,
        monotonic=lambda: 0.0,
    )

    remote.start(
        instance,
        phase="benchmark",
        max_seconds=3600,
        s3_prefix=orchestrator.config.s3_prefix,
    )

    assert [command[0] for command, _ in calls] == [
        "ssh",
        "ssh",
        "rsync",
        "ssh",
        "ssh",
    ]
    assert all(isinstance(command, list) for command, _ in calls)
    assert all("shell" not in kwargs for _, kwargs in calls)
    os_bootstrap = calls[1][1]["input"]
    dependency_bootstrap = calls[3][1]["input"]
    startup_script = calls[4][1]["input"]
    assert os_bootstrap.startswith("set -eu\n")
    assert "apt-get install" in os_bootstrap
    assert "rsync" in os_bootstrap
    assert "awscli-exe-linux-x86_64.zip" in os_bootstrap
    assert "uv\" python install 3.13" in dependency_bootstrap
    assert 'requirements.txt" --editable "$repo"' in dependency_bootstrap
    assert "import marioai, torch" in dependency_bootstrap
    assert "aws sts get-caller-identity" in dependency_bootstrap
    assert 'if ! kill -0 "$pid"' in startup_script
    assert "</dev/null" in startup_script


def test_ssh_readiness_retries_with_bounded_argument_vectors(
    orchestrator, tmp_path
):
    from scripts.aws_all32 import SshRemoteSupervisor

    ssh_key = tmp_path / "mario-training-key.pem"
    ssh_key.write_text("test-only", encoding="utf-8")
    local_repo = tmp_path / "MarioAI"
    local_repo.mkdir()
    clock = FakeMonotonic()
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) < 3:
            raise subprocess.CalledProcessError(
                255, command, stderr="connection refused"
            )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    remote = SshRemoteSupervisor(
        aws=orchestrator.aws,
        ssh_key=ssh_key,
        local_repo=local_repo,
        runner=runner,
        monotonic=clock,
        sleeper=lambda seconds: setattr(
            clock, "now", clock.now + seconds
        ),
    )

    remote._wait_for_ssh("ubuntu@203.0.113.10")

    assert len(calls) == 3
    assert clock.now == 10.0
    assert all(call[0][0] == "ssh" for call in calls)
    assert all("ConnectTimeout=10" in call[0] for call in calls)
    assert all(call[1]["timeout"] == 15 for call in calls)
    assert all("shell" not in call[1] for call in calls)


def test_remote_bootstrap_time_is_deducted_from_independent_timeout(
    orchestrator, tmp_path
):
    from scripts.aws_all32 import SshRemoteSupervisor

    orchestrator.preflight()
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    ssh_key = tmp_path / "mario-training-key.pem"
    ssh_key.write_text("test-only", encoding="utf-8")
    local_repo = tmp_path / "MarioAI"
    local_repo.mkdir()
    clock = FakeMonotonic()
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        clock.now += 10
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    remote = SshRemoteSupervisor(
        aws=orchestrator.aws,
        ssh_key=ssh_key,
        local_repo=local_repo,
        runner=runner,
        monotonic=clock,
        sleeper=lambda _seconds: None,
        startup_confirmation_seconds=1,
    )

    remote.start(
        instance,
        phase="benchmark",
        max_seconds=3600,
        s3_prefix=orchestrator.config.s3_prefix,
    )

    assert "3560" in calls[-1][0]


def test_startup_confirmation_rejects_immediate_background_failure(
    orchestrator, tmp_path
):
    from scripts.aws_all32 import SshRemoteSupervisor

    orchestrator.preflight()
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    ssh_key = tmp_path / "mario-training-key.pem"
    ssh_key.write_text("test-only", encoding="utf-8")
    local_repo = tmp_path / "source"
    local_repo.mkdir()
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    remote = SshRemoteSupervisor(
        aws=orchestrator.aws,
        ssh_key=ssh_key,
        local_repo=local_repo,
        runner=runner,
        monotonic=lambda: 0.0,
        startup_confirmation_seconds=1,
    )
    remote.start(
        instance,
        phase="benchmark",
        max_seconds=3600,
        s3_prefix=orchestrator.config.s3_prefix,
    )
    startup_script = calls[-1][1]["input"]
    execution_repo = tmp_path / "execution"
    (execution_repo / "scripts").mkdir(parents=True)
    failing_script = execution_repo / "scripts" / "cloud_train.sh"
    failing_script.write_text(
        "#!/usr/bin/env bash\necho immediate failure >&2\nexit 23\n",
        encoding="utf-8",
    )
    failing_script.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-s",
            "--",
            str(execution_repo),
            "benchmark",
            "30",
            str(execution_repo),
            "s3://test-bucket/test-prefix/",
        ],
        input=startup_script,
        check=False,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert completed.returncode == 23
    assert "immediate failure" in completed.stderr


def test_instance_readiness_timeout_is_bounded(orchestrator):
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    pending = {
        "instance_id": instance.instance_id,
        "instance_type": instance.instance_type,
        "public_ip": None,
        "state": "pending",
    }
    orchestrator.aws.project_results = [(pending,)] * 121

    with pytest.raises(
        AwsLifecycleError, match="readiness or absolute training deadline"
    ):
        orchestrator.wait_for_running_public_ip(instance)

    assert orchestrator.test_clock.now == 600.0


def test_ssh_remote_shutdown_waits_for_the_sync_trap(orchestrator, tmp_path):
    from scripts.aws_all32 import SshRemoteSupervisor

    orchestrator.preflight()
    instance = orchestrator.launch_guarded_instance(
        "benchmark", Decimal("1")
    )
    ssh_key = tmp_path / "mario-training-key.pem"
    ssh_key.write_text("test-only", encoding="utf-8")
    local_repo = tmp_path / "MarioAI"
    local_repo.mkdir()
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    remote = SshRemoteSupervisor(
        aws=orchestrator.aws,
        ssh_key=ssh_key,
        local_repo=local_repo,
        runner=runner,
    )

    remote.request_shutdown(instance)

    assert calls[0][0][0] == "ssh"
    assert 'while kill -0 "$pid"' in calls[0][1]["input"]


def _fake_executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\nset -u\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_cloud_supervisor_syncs_and_shuts_down_after_training_and_sync_failures(
    tmp_path,
):
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    command_log = tmp_path / "commands.log"
    repository = tmp_path / "MarioAI"
    (repository / "models").mkdir(parents=True)
    (repository / "reports").mkdir()
    logger = (
        'printf "%s" "$0" >> "$COMMAND_LOG"\n'
        'for argument in "$@"; do '
        'printf "\\t%s" "$argument" >> "$COMMAND_LOG"; done\n'
        'printf "\\n" >> "$COMMAND_LOG"\n'
    )
    _fake_executable(
        command_dir / "timeout",
        logger + 'exit "${TIMEOUT_EXIT:-7}"\n',
    )
    _fake_executable(
        command_dir / "aws",
        logger + 'exit "${AWS_EXIT:-0}"\n',
    )
    _fake_executable(command_dir / "sudo", logger + "exit 0\n")
    environment = {
        **os.environ,
        "PATH": f"{command_dir}:{os.environ['PATH']}",
        "COMMAND_LOG": str(command_log),
        "TIMEOUT_EXIT": "7",
        "AWS_EXIT": "1",
    }

    completed = subprocess.run(
        [
            "bash",
            str(CLOUD_TRAIN_PATH),
            "phase_1",
            "120",
            str(repository),
            "s3://bucket/marioai/all32/",
            "--resume",
            "models/previous.zip",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
        env=environment,
    )

    assert completed.returncode == 7
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands == [
        (
            f"{command_dir / 'timeout'}\t--signal=TERM\t--kill-after=300"
            "\t120\t.venv/bin/python\t-m\tmarioai.train"
            "\t--phase\tphase_1\t--resume\tmodels/previous.zip"
        ),
        (
            f"{command_dir / 'aws'}\ts3\tsync\t{repository}/models/"
            "\ts3://bucket/marioai/all32/models/"
            "\t--exclude\t*/latest.json"
        ),
        (
            f"{command_dir / 'aws'}\ts3\tsync\t{repository}/reports/"
            "\ts3://bucket/marioai/all32/reports/"
        ),
        f"{command_dir / 'sudo'}\tshutdown\t-h\tnow",
    ]


def test_cloud_supervisor_rejects_invalid_arguments_before_training(tmp_path):
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    command_log = tmp_path / "commands.log"
    _fake_executable(
        command_dir / "timeout",
        'printf "called\\n" >> "$COMMAND_LOG"\nexit 0\n',
    )
    environment = {
        **os.environ,
        "PATH": f"{command_dir}:{os.environ['PATH']}",
        "COMMAND_LOG": str(command_log),
    }

    completed = subprocess.run(
        ["bash", str(CLOUD_TRAIN_PATH), "phase_1", "not-seconds", "/tmp", "x"],
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
        env=environment,
    )

    assert completed.returncode != 0
    assert "maximum seconds" in completed.stderr
    assert not command_log.exists()


def test_cloud_supervisor_periodically_syncs_artifacts_then_manifest_and_final(
    tmp_path,
):
    """Catches missing 15-minute cadence or publishing latest before artifacts."""
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    command_log = tmp_path / "commands.log"
    repository = tmp_path / "MarioAI"
    checkpoint_dir = repository / "models" / "all32-phase_1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "ckpt_250000_steps.zip").write_bytes(b"model")
    (checkpoint_dir / "latest.json").write_text(
        '{"model":"ckpt_250000_steps.zip"}\n', encoding="utf-8"
    )
    (repository / "reports").mkdir()
    logger = (
        'printf "%s" "$0" >> "$COMMAND_LOG"\n'
        'for argument in "$@"; do '
        'printf "\\t%s" "$argument" >> "$COMMAND_LOG"; done\n'
        'printf "\\n" >> "$COMMAND_LOG"\n'
    )
    _fake_executable(
        command_dir / "timeout",
        logger + "sleep 2\nexit 7\n",
    )
    _fake_executable(command_dir / "aws", logger + "exit 0\n")
    _fake_executable(command_dir / "sudo", logger + "exit 0\n")
    environment = {
        **os.environ,
        "PATH": f"{command_dir}:{os.environ['PATH']}",
        "COMMAND_LOG": str(command_log),
        "MARIOAI_SYNC_INTERVAL_SECONDS": "1",
    }

    completed = subprocess.run(
        [
            "bash",
            str(CLOUD_TRAIN_PATH),
            "phase_1",
            "120",
            str(repository),
            "s3://bucket/marioai/all32/",
            "--run-name",
            "all32-phase_1",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
        env=environment,
    )

    assert completed.returncode == 7
    aws_calls = [
        line
        for line in command_log.read_text(encoding="utf-8").splitlines()
        if line.startswith(str(command_dir / "aws"))
    ]
    model_sync_indexes = [
        index
        for index, line in enumerate(aws_calls)
        if "\ts3\tsync\t" in line and "/models/" in line
    ]
    manifest_copy_indexes = [
        index
        for index, line in enumerate(aws_calls)
        if "\ts3\tcp\t" in line and line.endswith(
            "\ts3://bucket/marioai/all32/models/"
            "all32-phase_1/latest.json\t--only-show-errors"
        )
    ]
    assert len(model_sync_indexes) >= 2
    assert len(manifest_copy_indexes) >= 2
    assert all(
        sync_index < copy_index
        for sync_index, copy_index in zip(
            model_sync_indexes, manifest_copy_indexes, strict=True
        )
    )


def test_restore_downloads_only_exact_manifest_named_bundle(config, tmp_path):
    """Catches wildcard/latest guessing or downloading unmanifested S3 objects."""
    from scripts import aws_all32

    phase = "phase_1"
    run_name = "all32-phase_1"
    manifest_uri = (
        f"{config.s3_prefix}models/{run_name}/latest.json"
    )
    artifact_prefix = manifest_uri.removesuffix("latest.json")
    overrides = SimpleNamespace(
        levels=None,
        timesteps=None,
        n_envs=None,
        lr=None,
        ent_coef=None,
        level_weights_json=None,
    )
    run_config = training.load_training_config(
        "configs/all32.yaml", phase, overrides
    )
    signature = training.checkpoint_resume_signature(run_config, phase)
    artifacts = {
        "ckpt_250000_steps.zip": b"verified checkpoint",
        "ckpt_vecnormalize_250000_steps.pkl": b"paired normalization",
        "ckpt_signature_250000_steps.json": (
            json.dumps(signature, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode(),
        "ckpt_run_config_250000_steps.yaml": yaml.safe_dump(
            run_config, sort_keys=False
        ).encode(),
        "ckpt_budget_ledger_250000_steps.json": (
            json.dumps(
                {
                    "allocations": {
                        name: str(amount)
                        for name, amount in config.allocations.items()
                    },
                    "cap_usd": str(config.cap_usd),
                    "runs": [],
                    "spent_usd": "0",
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    }
    manifest = {
        "schema_version": 1,
        "run_name": run_name,
        "phase": phase,
        "num_timesteps": 250000,
        "model": "ckpt_250000_steps.zip",
        "sha256": hashlib.sha256(
            artifacts["ckpt_250000_steps.zip"]
        ).hexdigest(),
        "action_set": "complex",
        "action_count": 12,
        "extractor": "impala",
        "extractor_class": (
            "marioai.features.ImpalaCnnFeaturesExtractor"
        ),
        "normalize_reward": True,
        "vecnormalize": "ckpt_vecnormalize_250000_steps.pkl",
        "vecnormalize_sha256": hashlib.sha256(
            artifacts["ckpt_vecnormalize_250000_steps.pkl"]
        ).hexdigest(),
        "signature": "ckpt_signature_250000_steps.json",
        "signature_sha256": hashlib.sha256(
            artifacts["ckpt_signature_250000_steps.json"]
        ).hexdigest(),
        "run_config": "ckpt_run_config_250000_steps.yaml",
        "run_config_sha256": hashlib.sha256(
            artifacts["ckpt_run_config_250000_steps.yaml"]
        ).hexdigest(),
        "budget_ledger": "ckpt_budget_ledger_250000_steps.json",
        "budget_ledger_sha256": hashlib.sha256(
            artifacts["ckpt_budget_ledger_250000_steps.json"]
        ).hexdigest(),
    }
    objects = {
        manifest_uri: (
            json.dumps(manifest, sort_keys=True) + "\n"
        ).encode(),
        **{
            f"{artifact_prefix}{name}": content
            for name, content in artifacts.items()
        },
        f"{artifact_prefix}untrusted.zip": b"must not download",
    }

    class FakeCheckpointStore:
        def __init__(self):
            self.downloads = []

        def download(self, uri, destination):
            self.downloads.append(uri)
            Path(destination).write_bytes(objects[uri])

    store = FakeCheckpointStore()
    ledger_path = tmp_path / "aws-spend.json"
    BudgetLedger(
        cap_usd=config.cap_usd, allocations=config.allocations
    ).save(ledger_path)

    bundle = aws_all32.restore_checkpoint_bundle(
        config=config,
        phase=phase,
        checkpoint_s3_uri=manifest_uri,
        repo_dir=tmp_path,
        ledger_path=ledger_path,
        object_store=store,
        model_validator=lambda path, cfg: (
            path.name,
            cfg["env"]["action_set"],
        ),
    )

    assert bundle.model_path.name == manifest["model"]
    assert training.sha256_file(bundle.model_path) == manifest["sha256"]
    assert store.downloads == [
        manifest_uri,
        *[
            f"{artifact_prefix}{manifest[field]}"
            for field in (
                "model",
                "run_config",
                "signature",
                "vecnormalize",
                "budget_ledger",
            )
        ],
    ]


def test_cli_resume_verifies_before_paid_launch_and_passes_exact_paths(
    config, tmp_path, monkeypatch
):
    """Catches launching first or letting remote training guess resume files."""
    from scripts import aws_all32

    staging = tmp_path / ".resume" / "phase_1-test"
    staging.mkdir(parents=True)
    paths = {
        name: staging / filename
        for name, filename in {
            "manifest": "latest.json",
            "model": "ckpt_250000_steps.zip",
            "run_config": "ckpt_run_config_250000_steps.yaml",
            "signature": "ckpt_signature_250000_steps.json",
            "vecnormalize": "ckpt_vecnormalize_250000_steps.pkl",
            "budget_ledger": "ckpt_budget_ledger_250000_steps.json",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"test")
    bundle = aws_all32.ResumeBundle(
        root=staging,
        manifest_path=paths["manifest"],
        model_path=paths["model"],
        run_config_path=paths["run_config"],
        signature_path=paths["signature"],
        vecnormalize_path=paths["vecnormalize"],
        budget_ledger_path=paths["budget_ledger"],
        manifest={
            "run_name": "all32-phase_1",
            "model": paths["model"].name,
            "sha256": hashlib.sha256(b"test").hexdigest(),
            "run_config": paths["run_config"].name,
            "run_config_sha256": hashlib.sha256(b"test").hexdigest(),
            "signature": paths["signature"].name,
            "signature_sha256": hashlib.sha256(b"test").hexdigest(),
            "vecnormalize": paths["vecnormalize"].name,
            "vecnormalize_sha256": hashlib.sha256(b"test").hexdigest(),
            "budget_ledger": paths["budget_ledger"].name,
            "budget_ledger_sha256": hashlib.sha256(b"test").hexdigest(),
        },
    )
    clock = FakeMonotonic()
    aws = FakeLifecycleAws(config)
    remote = FakeRemote(clock)
    restore_events = []

    def fake_restore(**kwargs):
        assert not aws.run_instances_called
        restore_events.append(kwargs["checkpoint_s3_uri"])
        return bundle

    monkeypatch.setattr(
        aws_all32, "restore_checkpoint_bundle", fake_restore
    )
    ledger_path = tmp_path / "aws-spend.json"

    result = aws_all32.main(
        [
            "resume",
            "--config",
            str(CONFIG_PATH),
            "--ledger",
            str(ledger_path),
            "--phase",
            "phase_1",
            "--max-hours",
            "1",
            "--repo-dir",
            str(tmp_path),
            "--checkpoint-s3-uri",
            (
                f"{config.s3_prefix}models/all32-phase_1/latest.json"
            ),
        ],
        stdout=io.StringIO(),
        aws_override=aws,
        remote=remote,
        checkpoint_store=object(),
        monotonic=clock,
        sleeper=lambda _seconds: None,
        client_token_factory=lambda: "resume-idempotency-token",
    )

    assert result == 0
    assert restore_events == [
        f"{config.s3_prefix}models/all32-phase_1/latest.json"
    ]
    assert remote.training_args == [
        (
            "--config",
            "configs/all32.yaml",
            "--run-name",
            "all32-phase_1",
            "--resume",
            ".resume/phase_1-test/ckpt_250000_steps.zip",
            "--resume-vecnormalize",
            (
                ".resume/phase_1-test/"
                "ckpt_vecnormalize_250000_steps.pkl"
            ),
            "--budget-ledger-snapshot",
            (
                ".resume/phase_1-test/"
                "ckpt_budget_ledger_250000_steps.json"
            ),
        )
    ]


@pytest.mark.parametrize(
    "manifest_uri",
    [
        "s3://other-bucket/marioai/all32/run/latest.json",
        (
            "s3://defectlens-phase3-002559670021/"
            "outside/all32/run/latest.json"
        ),
        (
            "s3://defectlens-phase3-002559670021/marioai/all32/"
            "%2e%2e/run/latest.json"
        ),
        (
            "s3://defectlens-phase3-002559670021/marioai/all32/"
            "run/not-latest.json"
        ),
    ],
)
def test_restore_rejects_unconfined_manifest_before_download(
    config, tmp_path, manifest_uri
):
    """Catches reading attacker-selected objects outside the approved prefix."""
    from scripts.aws_all32 import (
        AwsLifecycleError,
        restore_checkpoint_bundle,
    )

    class NoDownload:
        def download(self, _uri, _destination):
            raise AssertionError("unconfined URI reached the S3 boundary")

    with pytest.raises(AwsLifecycleError, match="S3|prefix"):
        restore_checkpoint_bundle(
            config=config,
            phase="phase_1",
            checkpoint_s3_uri=manifest_uri,
            repo_dir=tmp_path,
            ledger_path=tmp_path / "ledger.json",
            object_store=NoDownload(),
            model_validator=lambda _path, _cfg: None,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "../ckpt_250000_steps.zip",
        "/tmp/ckpt_250000_steps.zip",
        "nested/ckpt_250000_steps.zip",
        r"..\ckpt_250000_steps.zip",
        "%2e%2e.zip",
    ],
)
def test_restore_rejects_untrusted_artifact_name_without_fetching_it(
    config, tmp_path, model_name
):
    """Catches path traversal and S3-prefix escape through manifest filenames."""
    from scripts.aws_all32 import (
        AwsLifecycleError,
        restore_checkpoint_bundle,
    )

    uri = f"{config.s3_prefix}models/run/latest.json"
    manifest = {
        "schema_version": 1,
        "run_name": "run",
        "phase": "phase_1",
        "num_timesteps": 250000,
        "model": model_name,
        "sha256": "0" * 64,
        "action_set": "complex",
        "action_count": 12,
        "extractor": "impala",
        "extractor_class": (
            "marioai.features.ImpalaCnnFeaturesExtractor"
        ),
        "normalize_reward": True,
        "vecnormalize": "ckpt_vecnormalize_250000_steps.pkl",
        "vecnormalize_sha256": "0" * 64,
        "signature": "ckpt_signature_250000_steps.json",
        "signature_sha256": "0" * 64,
        "run_config": "ckpt_run_config_250000_steps.yaml",
        "run_config_sha256": "0" * 64,
        "budget_ledger": "ckpt_budget_ledger_250000_steps.json",
        "budget_ledger_sha256": "0" * 64,
    }

    class ManifestOnlyStore:
        def __init__(self):
            self.downloads = []

        def download(self, object_uri, destination):
            self.downloads.append(object_uri)
            if object_uri != uri:
                raise AssertionError("unsafe artifact was fetched")
            Path(destination).write_text(
                json.dumps(manifest), encoding="utf-8"
            )

    store = ManifestOnlyStore()
    with pytest.raises(AwsLifecycleError, match="filename"):
        restore_checkpoint_bundle(
            config=config,
            phase="phase_1",
            checkpoint_s3_uri=uri,
            repo_dir=tmp_path,
            ledger_path=tmp_path / "ledger.json",
            object_store=store,
            model_validator=lambda _path, _cfg: None,
        )
    assert store.downloads == [uri]


def test_restore_rejects_hash_mismatch_before_model_deserialization(
    config, tmp_path
):
    """Catches trusting a manifest filename without authenticating its bytes."""
    from scripts.aws_all32 import (
        AwsLifecycleError,
        restore_checkpoint_bundle,
    )

    uri = f"{config.s3_prefix}models/run/latest.json"
    manifest = {
        "schema_version": 1,
        "run_name": "run",
        "phase": "phase_1",
        "num_timesteps": 250000,
        "model": "ckpt_250000_steps.zip",
        "sha256": "0" * 64,
        "action_set": "complex",
        "action_count": 12,
        "extractor": "impala",
        "extractor_class": (
            "marioai.features.ImpalaCnnFeaturesExtractor"
        ),
        "normalize_reward": True,
        "vecnormalize": "ckpt_vecnormalize_250000_steps.pkl",
        "vecnormalize_sha256": "0" * 64,
        "signature": "ckpt_signature_250000_steps.json",
        "signature_sha256": "0" * 64,
        "run_config": "ckpt_run_config_250000_steps.yaml",
        "run_config_sha256": "0" * 64,
        "budget_ledger": "ckpt_budget_ledger_250000_steps.json",
        "budget_ledger_sha256": "0" * 64,
    }
    model_validator_called = False

    class TamperedStore:
        def download(self, object_uri, destination):
            if object_uri == uri:
                Path(destination).write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
            else:
                Path(destination).write_bytes(b"tampered")

    def validate_model(_path, _cfg):
        nonlocal model_validator_called
        model_validator_called = True

    with pytest.raises(AwsLifecycleError, match="SHA-256"):
        restore_checkpoint_bundle(
            config=config,
            phase="phase_1",
            checkpoint_s3_uri=uri,
            repo_dir=tmp_path,
            ledger_path=tmp_path / "ledger.json",
            object_store=TamperedStore(),
            model_validator=validate_model,
        )
    assert not model_validator_called


def test_resume_rehash_failure_at_mutation_boundary_prevents_launch(
    orchestrator, tmp_path
):
    """Catches a verified checkpoint being swapped during final AWS preflight."""
    from scripts.aws_all32 import (
        AwsLifecycleError,
        ResumeBundle,
        verify_resume_bundle,
    )

    artifact_names = {
        "model": "ckpt_250000_steps.zip",
        "run_config": "ckpt_run_config_250000_steps.yaml",
        "signature": "ckpt_signature_250000_steps.json",
        "budget_ledger": "ckpt_budget_ledger_250000_steps.json",
    }
    paths = {}
    manifest = {"run_name": "all32-phase_1"}
    for field, filename in artifact_names.items():
        path = tmp_path / filename
        path.write_bytes(b"verified")
        paths[field] = path
        manifest[field] = filename
        manifest[
            "sha256" if field == "model" else f"{field}_sha256"
        ] = hashlib.sha256(b"verified").hexdigest()
    manifest_path = tmp_path / "latest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    bundle = ResumeBundle(
        root=tmp_path,
        manifest_path=manifest_path,
        model_path=paths["model"],
        run_config_path=paths["run_config"],
        signature_path=paths["signature"],
        vecnormalize_path=None,
        budget_ledger_path=paths["budget_ledger"],
        manifest=manifest,
    )

    def recheck_after_swap():
        paths["model"].write_bytes(b"swapped")
        verify_resume_bundle(bundle)

    with pytest.raises(AwsLifecycleError, match="model changed"):
        orchestrator.launch_guarded_instance(
            "phase_1",
            Decimal("1"),
            before_mutation=recheck_after_swap,
        )
    assert not orchestrator.aws.run_instances_called


def test_s3_checkpoint_store_uses_one_fixed_read_only_argument_vector(
    config, tmp_path
):
    """Catches shell interpolation, wildcard sync, or profile/region override."""
    from scripts.aws_all32 import S3CheckpointStore

    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[4]).write_bytes(b"manifest")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    store = S3CheckpointStore(
        profile=config.profile,
        region=config.region,
        runner=runner,
    )
    destination = tmp_path / "latest.json"
    uri = f"{config.s3_prefix}models/run/latest.json"

    store.download(uri, destination)

    assert destination.read_bytes() == b"manifest"
    command, kwargs = calls[0]
    assert command[:4] == ["aws", "s3", "cp", uri]
    assert command[5:] == [
        "--only-show-errors",
        "--no-progress",
        "--profile",
        config.profile,
        "--region",
        config.region,
    ]
    assert kwargs == {
        "check": True,
        "text": True,
        "capture_output": True,
        "timeout": 60,
    }
