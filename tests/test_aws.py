from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from marioai.aws import (
    AwsCli,
    AwsCliError,
    AwsConfig,
    AwsPreflightError,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "aws-all32.yaml"
SUBNET_IDS = (
    "subnet-0ba0242531d6615f3",
    "subnet-0870eed3fd2b7dfab",
    "subnet-02e9c5f8deb69ad62",
    "subnet-0e5d82120140b0b4e",
    "subnet-02ab8427ae66e6e36",
    "subnet-0fa9ec503414b2be9",
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


@pytest.fixture
def config() -> AwsConfig:
    return AwsConfig.from_yaml(CONFIG_PATH)


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
                "InstanceProfileName": config.instance_profile,
                "Arn": (
                    f"arn:aws:iam::{config.account_id}:instance-profile/"
                    f"{config.instance_profile}"
                ),
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
    return runner


def _preflight_cli(config: AwsConfig) -> tuple[AwsCli, FakeRunner]:
    runner = _successful_runner(config)
    cli = AwsCli(profile=config.profile, region=config.region, runner=runner)
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
    assert [call[0][1:3] for call in runner.calls] == [
        ["sts", "get-caller-identity"],
        ["ec2", "describe-vpcs"],
        ["ec2", "describe-subnets"],
        ["ec2", "describe-security-groups"],
        ["ec2", "describe-key-pairs"],
        ["iam", "get-instance-profile"],
        ["s3api", "list-objects-v2"],
        ["ec2", "describe-instances"],
    ]


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
        [
            "ec2",
            "describe-spot-price-history",
            "--instance-types",
            "c7i.16xlarge",
            "c7i.8xlarge",
            "--product-descriptions",
            "Linux/UNIX",
        ],
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


def test_spot_selection_rejects_malformed_allowed_offer(config):
    cli, runner = _preflight_cli(config)
    cli.preflight(config)
    runner.add(
        [
            "ec2",
            "describe-spot-price-history",
            "--instance-types",
            "c7i.8xlarge",
            "--product-descriptions",
            "Linux/UNIX",
        ],
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
