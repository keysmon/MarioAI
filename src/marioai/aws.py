"""Strictly read-only AWS configuration checks for all-32 training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import subprocess
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import yaml


class AwsCliError(RuntimeError):
    """Raised when the local AWS CLI boundary cannot return valid JSON."""


class AwsPreflightError(RuntimeError):
    """Raised when AWS does not match the approved training configuration."""


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _decimal_string(value: Any, field_name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be stored as a string")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field_name} must be a valid Decimal") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field_name} must be a finite non-negative Decimal")
    return result


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    result = tuple(_required_string(item, field_name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _decimal_mapping(value: Any, field_name: str) -> Mapping[str, Decimal]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    result = {
        _required_string(key, f"{field_name} key"): _decimal_string(
            item, f"{field_name}.{key}"
        )
        for key, item in value.items()
    }
    return MappingProxyType(result)


@dataclass(frozen=True)
class AwsConfig:
    """Validated AWS resources and budget constraints for one account."""

    profile: str
    account_id: str
    region: str
    ami_ssm_parameter: str
    vpc_id: str
    subnet_ids: tuple[str, ...]
    security_group_id: str
    key_name: str
    instance_profile: str
    s3_prefix: str
    instance_types: tuple[str, ...]
    on_demand_ceiling_usd: Mapping[str, Decimal]
    root_volume_gb: int
    gp3_monthly_usd_per_gb: Decimal
    cap_usd: Decimal
    shutdown_threshold_usd: Decimal
    grace_minutes: int
    allocations: Mapping[str, Decimal]

    @classmethod
    def from_yaml(cls, path: Path) -> AwsConfig:
        """Load a config without accepting implicit float-to-Decimal coercion."""
        path = Path(path)
        try:
            with path.open(encoding="utf-8") as config_file:
                raw = yaml.safe_load(config_file)
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"cannot load AWS configuration {path}") from error
        if not isinstance(raw, Mapping):
            raise ValueError("AWS configuration must be a mapping")

        expected_fields = {
            "profile",
            "account_id",
            "region",
            "ami_ssm_parameter",
            "vpc_id",
            "subnet_ids",
            "security_group_id",
            "key_name",
            "instance_profile",
            "s3_prefix",
            "instance_types",
            "on_demand_ceiling_usd",
            "root_volume_gb",
            "gp3_monthly_usd_per_gb",
            "cap_usd",
            "shutdown_threshold_usd",
            "grace_minutes",
            "allocations",
        }
        missing = sorted(expected_fields - raw.keys())
        unexpected = sorted(raw.keys() - expected_fields)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected fields: {', '.join(unexpected)}")
            raise ValueError(f"invalid AWS configuration ({'; '.join(details)})")

        config = cls(
            profile=_required_string(raw["profile"], "profile"),
            account_id=_required_string(raw["account_id"], "account_id"),
            region=_required_string(raw["region"], "region"),
            ami_ssm_parameter=_required_string(
                raw["ami_ssm_parameter"], "ami_ssm_parameter"
            ),
            vpc_id=_required_string(raw["vpc_id"], "vpc_id"),
            subnet_ids=_string_tuple(raw["subnet_ids"], "subnet_ids"),
            security_group_id=_required_string(
                raw["security_group_id"], "security_group_id"
            ),
            key_name=_required_string(raw["key_name"], "key_name"),
            instance_profile=_required_string(
                raw["instance_profile"], "instance_profile"
            ),
            s3_prefix=_required_string(raw["s3_prefix"], "s3_prefix"),
            instance_types=_string_tuple(raw["instance_types"], "instance_types"),
            on_demand_ceiling_usd=_decimal_mapping(
                raw["on_demand_ceiling_usd"], "on_demand_ceiling_usd"
            ),
            root_volume_gb=_required_int(raw["root_volume_gb"], "root_volume_gb"),
            gp3_monthly_usd_per_gb=_decimal_string(
                raw["gp3_monthly_usd_per_gb"], "gp3_monthly_usd_per_gb"
            ),
            cap_usd=_decimal_string(raw["cap_usd"], "cap_usd"),
            shutdown_threshold_usd=_decimal_string(
                raw["shutdown_threshold_usd"], "shutdown_threshold_usd"
            ),
            grace_minutes=_required_int(raw["grace_minutes"], "grace_minutes"),
            allocations=_decimal_mapping(raw["allocations"], "allocations"),
        )
        config._validate_relationships()
        return config

    def _validate_relationships(self) -> None:
        if len(self.account_id) != 12 or not self.account_id.isdigit():
            raise ValueError("account_id must contain exactly 12 digits")
        if set(self.on_demand_ceiling_usd) != set(self.instance_types):
            raise ValueError(
                "on_demand_ceiling_usd must cover exactly the instance_types"
            )
        if self.shutdown_threshold_usd >= self.cap_usd:
            raise ValueError("shutdown_threshold_usd must be below cap_usd")
        if sum(self.allocations.values(), Decimal("0")) != self.cap_usd:
            raise ValueError("allocations must sum exactly to cap_usd")
        _parse_s3_prefix(self.s3_prefix)


@dataclass(frozen=True)
class PreflightResult:
    """Verified resource identity returned by a successful preflight."""

    account_id: str
    vpc_id: str
    subnet_azs: tuple[tuple[str, str], ...]
    s3_bucket: str
    s3_key_prefix: str
    running_project_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class SpotOffer:
    """One latest Spot price in an allowed subnet and availability zone."""

    instance_type: str
    availability_zone: str
    subnet_id: str
    hourly_usd: Decimal
    timestamp: datetime


_READ_ONLY_OPERATIONS = frozenset(
    {
        ("sts", "get-caller-identity"),
        ("ec2", "describe-vpcs"),
        ("ec2", "describe-subnets"),
        ("ec2", "describe-security-groups"),
        ("ec2", "describe-key-pairs"),
        ("ec2", "describe-instances"),
        ("ec2", "describe-spot-price-history"),
        ("iam", "get-instance-profile"),
        ("s3api", "list-objects-v2"),
    }
)
_FIXED_GLOBAL_OPTIONS = frozenset(
    {"--profile", "--region", "--output", "--endpoint-url"}
)
_SPOT_MAX_AGE = timedelta(days=7)
_SPOT_MAX_FUTURE_SKEW = timedelta(minutes=5)
_AWS_CLI_TIMEOUT_SECONDS = 60


class AwsCli:
    """Small AWS CLI adapter with no state-changing operation available."""

    def __init__(
        self,
        profile: str,
        region: str,
        *,
        runner: Callable[..., Any] = subprocess.run,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.profile = _required_string(profile, "profile")
        self.region = _required_string(region, "region")
        self._runner = runner
        self._clock = clock
        self._preflight_config: AwsConfig | None = None
        self._subnet_by_az: dict[str, str] = {}

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: int | float | None = None,
    ) -> dict | list | str:
        """Run one explicitly allowed read-only operation and parse strict JSON."""
        if isinstance(args, (str, bytes)) or len(args) < 2:
            raise AwsCliError("AWS arguments must name a service and operation")
        if not all(isinstance(argument, str) and argument for argument in args):
            raise AwsCliError("AWS arguments must be non-empty strings")
        operation = (args[0], args[1])
        if operation not in _READ_ONLY_OPERATIONS:
            raise AwsCliError(
                f"{' '.join(operation)} is not an allowed read-only operation"
            )
        fixed_override = next(
            (
                argument
                for argument in args[2:]
                if any(
                    argument == option or argument.startswith(f"{option}=")
                    for option in _FIXED_GLOBAL_OPTIONS
                )
            ),
            None,
        )
        if fixed_override is not None:
            raise AwsCliError(f"{fixed_override} is controlled by AwsCli")
        timeout = (
            _AWS_CLI_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not Decimal(str(timeout)).is_finite()
            or timeout <= 0
            or timeout > _AWS_CLI_TIMEOUT_SECONDS
        ):
            raise AwsCliError(
                "AWS timeout must be positive and no greater than 60 seconds"
            )

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
        operation_name = " ".join(operation)
        try:
            completed = self._runner(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise AwsCliError(
                f"AWS {operation_name} timed out after "
                f"{timeout} seconds"
            ) from error
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise AwsCliError(f"AWS {operation_name} failed{detail}") from error
        except OSError as error:
            raise AwsCliError(
                f"could not execute AWS CLI for {operation_name}: {error}"
            ) from error

        stdout = completed.stdout
        if not isinstance(stdout, str):
            raise AwsCliError(f"AWS {operation_name} returned non-text output")
        if not stdout.strip():
            return ""
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise AwsCliError(
                f"AWS {operation_name} returned invalid JSON"
            ) from error
        if not isinstance(payload, (dict, list, str)):
            raise AwsCliError(
                f"AWS {operation_name} returned an unsupported JSON value"
            )
        return payload

    def preflight(self, config: AwsConfig) -> PreflightResult:
        """Verify exact prerequisites using describe/get/list operations only."""
        self._preflight_config = None
        self._subnet_by_az = {}
        if not isinstance(config, AwsConfig):
            raise AwsPreflightError("preflight requires an AwsConfig")
        if self.profile != config.profile:
            raise AwsPreflightError(
                f"CLI profile {self.profile!r} does not match {config.profile!r}"
            )
        if self.region != config.region:
            raise AwsPreflightError(
                f"CLI region {self.region!r} does not match {config.region!r}"
            )

        identity = self._preflight_run(
            "caller identity", ["sts", "get-caller-identity"]
        )
        identity_mapping = _response_mapping(identity, "caller identity")
        if identity_mapping.get("Account") != config.account_id:
            raise AwsPreflightError(
                "expected AWS account "
                f"{config.account_id}, got {identity_mapping.get('Account')!r}"
            )

        vpcs = _response_list(
            self._preflight_run(
                "VPC", ["ec2", "describe-vpcs", "--vpc-ids", config.vpc_id]
            ),
            "Vpcs",
            "VPC",
        )
        if len(vpcs) != 1 or vpcs[0].get("VpcId") != config.vpc_id:
            raise AwsPreflightError(f"expected VPC {config.vpc_id}")
        if vpcs[0].get("State") != "available":
            raise AwsPreflightError(f"VPC {config.vpc_id} is not available")

        subnets = _response_list(
            self._preflight_run(
                "subnets",
                [
                    "ec2",
                    "describe-subnets",
                    "--subnet-ids",
                    *config.subnet_ids,
                ],
            ),
            "Subnets",
            "subnets",
        )
        expected_subnets = set(config.subnet_ids)
        returned_subnet_ids: list[str] = []
        for subnet in subnets:
            subnet_id = subnet.get("SubnetId")
            if not isinstance(subnet_id, str) or not subnet_id:
                raise AwsPreflightError(
                    "subnets response contains an invalid SubnetId"
                )
            returned_subnet_ids.append(subnet_id)
        returned_subnets = set(returned_subnet_ids)
        if returned_subnets != expected_subnets or len(subnets) != len(
            config.subnet_ids
        ):
            raise AwsPreflightError(
                "subnets returned by AWS do not exactly match configured subnet IDs"
            )
        subnet_by_id = {subnet["SubnetId"]: subnet for subnet in subnets}
        subnet_azs: list[tuple[str, str]] = []
        used_azs: set[str] = set()
        for subnet_id in config.subnet_ids:
            subnet = subnet_by_id[subnet_id]
            if subnet.get("VpcId") != config.vpc_id:
                raise AwsPreflightError(
                    f"subnets must belong to expected VPC {config.vpc_id}"
                )
            if subnet.get("State") != "available":
                raise AwsPreflightError(f"subnet {subnet_id} is not available")
            availability_zone = subnet.get("AvailabilityZone")
            if (
                not isinstance(availability_zone, str)
                or re.fullmatch(
                    rf"{re.escape(config.region)}[a-z]", availability_zone
                )
                is None
            ):
                raise AwsPreflightError(
                    f"subnet {subnet_id} has an invalid availability zone"
                )
            if availability_zone in used_azs:
                raise AwsPreflightError(
                    "configured subnets must map to distinct availability zones"
                )
            used_azs.add(availability_zone)
            subnet_azs.append((subnet_id, availability_zone))

        security_groups = _response_list(
            self._preflight_run(
                "security group",
                [
                    "ec2",
                    "describe-security-groups",
                    "--group-ids",
                    config.security_group_id,
                ],
            ),
            "SecurityGroups",
            "security group",
        )
        if (
            len(security_groups) != 1
            or security_groups[0].get("GroupId") != config.security_group_id
            or security_groups[0].get("VpcId") != config.vpc_id
        ):
            raise AwsPreflightError(
                f"security group {config.security_group_id} is not in "
                f"expected VPC {config.vpc_id}"
            )

        key_pairs = _response_list(
            self._preflight_run(
                "key pair",
                [
                    "ec2",
                    "describe-key-pairs",
                    "--key-names",
                    config.key_name,
                ],
            ),
            "KeyPairs",
            "key pair",
        )
        if len(key_pairs) != 1 or key_pairs[0].get("KeyName") != config.key_name:
            raise AwsPreflightError(f"expected key pair {config.key_name}")

        profile_payload = _response_mapping(
            self._preflight_run(
                "instance profile",
                [
                    "iam",
                    "get-instance-profile",
                    "--instance-profile-name",
                    config.instance_profile,
                ],
            ),
            "instance profile",
        )
        instance_profile = profile_payload.get("InstanceProfile")
        if (
            not isinstance(instance_profile, Mapping)
            or instance_profile.get("InstanceProfileName")
            != config.instance_profile
        ):
            raise AwsPreflightError(
                f"expected instance profile {config.instance_profile}"
            )

        bucket, key_prefix = _parse_s3_prefix(config.s3_prefix)
        s3_list = _response_mapping(
            self._preflight_run(
                "S3 prefix list access",
                [
                    "s3api",
                    "list-objects-v2",
                    "--bucket",
                    bucket,
                    "--prefix",
                    key_prefix,
                    "--max-items",
                    "1",
                ],
            ),
            "S3 prefix list",
        )
        key_count = s3_list.get("KeyCount")
        if (
            isinstance(key_count, bool)
            or not isinstance(key_count, int)
            or key_count < 0
        ):
            raise AwsPreflightError(
                "S3 prefix list response has invalid KeyCount"
            )

        reservations = _response_list(
            self._preflight_run(
                "running project instances",
                [
                    "ec2",
                    "describe-instances",
                    "--filters",
                    "Name=tag:Project,Values=MarioAI-All32",
                    "Name=instance-state-name,Values=running",
                ],
            ),
            "Reservations",
            "running project instances",
        )
        running_instance_ids: list[str] = []
        for reservation in reservations:
            instances = reservation.get("Instances")
            if not isinstance(instances, list):
                raise AwsPreflightError(
                    "running project instances response has invalid Instances"
                )
            for instance in instances:
                if not isinstance(instance, Mapping):
                    raise AwsPreflightError(
                        "running project instances response is malformed"
                    )
                instance_id = instance.get("InstanceId")
                if not isinstance(instance_id, str) or not instance_id:
                    raise AwsPreflightError(
                        "running project instance has no valid InstanceId"
                    )
                running_instance_ids.append(instance_id)
        if running_instance_ids:
            raise AwsPreflightError(
                "running Project=MarioAI-All32 instance(s) already exist: "
                + ", ".join(sorted(running_instance_ids))
            )

        subnet_by_az = {
            availability_zone: subnet_id
            for subnet_id, availability_zone in subnet_azs
        }
        current_offers = self._latest_spot_prices(
            config.instance_types, config, subnet_by_az
        )
        if not current_offers:
            raise AwsPreflightError(
                "at least one current allowed Spot offer is required"
            )

        self._preflight_config = config
        self._subnet_by_az = subnet_by_az
        return PreflightResult(
            account_id=config.account_id,
            vpc_id=config.vpc_id,
            subnet_azs=tuple(subnet_azs),
            s3_bucket=bucket,
            s3_key_prefix=key_prefix,
            running_project_instance_ids=(),
        )

    def latest_spot_prices(
        self, instance_types: Sequence[str]
    ) -> tuple[SpotOffer, ...]:
        """Return latest allowed offers sorted by price and stable tie-breakers."""
        config = self._preflight_config
        if config is None:
            raise AwsPreflightError(
                "successful preflight is required before Spot price lookup"
            )
        return self._latest_spot_prices(
            instance_types, config, self._subnet_by_az
        )

    def _latest_spot_prices(
        self,
        instance_types: Sequence[str],
        config: AwsConfig,
        subnet_by_az: Mapping[str, str],
    ) -> tuple[SpotOffer, ...]:
        if isinstance(instance_types, (str, bytes)):
            raise AwsPreflightError("instance_types must be a sequence")
        raw_types = tuple(instance_types)
        if not raw_types or not all(
            isinstance(instance_type, str) and instance_type
            for instance_type in raw_types
        ):
            raise AwsPreflightError(
                "instance_types must contain non-empty strings"
            )
        requested_types = tuple(sorted(set(raw_types)))
        disallowed = set(requested_types) - set(config.instance_types)
        if disallowed:
            raise AwsPreflightError(
                "instance types outside configuration are not allowed: "
                + ", ".join(sorted(disallowed))
            )

        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise AwsPreflightError(
                "Spot price clock must return a timezone-aware datetime"
            )
        now = now.astimezone(timezone.utc)
        history = _response_list(
            self._preflight_run(
                "Spot price history",
                [
                    "ec2",
                    "describe-spot-price-history",
                    "--instance-types",
                    *requested_types,
                    "--product-descriptions",
                    "Linux/UNIX",
                    "--start-time",
                    (now - _SPOT_MAX_AGE).isoformat(),
                    "--end-time",
                    now.isoformat(),
                ],
            ),
            "SpotPriceHistory",
            "Spot price history",
        )
        latest: dict[tuple[str, str], SpotOffer] = {}
        saw_out_of_window_offer = False
        for item in history:
            instance_type = item.get("InstanceType")
            availability_zone = item.get("AvailabilityZone")
            if not isinstance(instance_type, str) or not isinstance(
                availability_zone, str
            ):
                raise AwsPreflightError(
                    "malformed Spot offer instance type or availability zone"
                )
            if (
                instance_type not in requested_types
                or availability_zone not in subnet_by_az
            ):
                continue
            try:
                price_text = item["SpotPrice"]
                timestamp_text = item["Timestamp"]
                if not isinstance(price_text, str) or not isinstance(
                    timestamp_text, str
                ):
                    raise ValueError
                hourly_usd = Decimal(price_text)
                timestamp = datetime.fromisoformat(
                    timestamp_text.replace("Z", "+00:00")
                )
                if (
                    not hourly_usd.is_finite()
                    or hourly_usd <= 0
                    or timestamp.tzinfo is None
                ):
                    raise ValueError
            except (KeyError, InvalidOperation, ValueError) as error:
                raise AwsPreflightError(
                    "malformed Spot offer for allowed instance type/AZ"
                ) from error
            if (
                timestamp < now - _SPOT_MAX_AGE
                or timestamp > now + _SPOT_MAX_FUTURE_SKEW
            ):
                saw_out_of_window_offer = True
                continue
            offer = SpotOffer(
                instance_type=instance_type,
                availability_zone=availability_zone,
                subnet_id=subnet_by_az[availability_zone],
                hourly_usd=hourly_usd,
                timestamp=timestamp,
            )
            key = (instance_type, availability_zone)
            previous = latest.get(key)
            if previous is None or (offer.timestamp, -offer.hourly_usd) > (
                previous.timestamp,
                -previous.hourly_usd,
            ):
                latest[key] = offer

        if not latest and saw_out_of_window_offer:
            raise AwsPreflightError(
                "allowed Spot offers fall outside the current freshness window"
            )
        return tuple(
            sorted(
                latest.values(),
                key=lambda offer: (
                    offer.hourly_usd,
                    offer.instance_type,
                    offer.availability_zone,
                    offer.subnet_id,
                ),
            )
        )

    def _preflight_run(
        self, description: str, args: Sequence[str]
    ) -> dict | list | str:
        try:
            return self.run(args)
        except AwsCliError as error:
            raise AwsPreflightError(
                f"cannot verify {description}: {error}"
            ) from error


def _response_mapping(payload: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise AwsPreflightError(f"{description} response must be a JSON object")
    return payload


def _response_list(
    payload: Any, field_name: str, description: str
) -> list[Mapping[str, Any]]:
    response = _response_mapping(payload, description)
    values = response.get(field_name)
    if not isinstance(values, list) or not all(
        isinstance(value, Mapping) for value in values
    ):
        raise AwsPreflightError(
            f"{description} response has invalid {field_name}"
        )
    return values


def _parse_s3_prefix(value: str) -> tuple[str, str]:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or not parsed.path.endswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("s3_prefix must be an s3:// bucket prefix ending in /")
    key_prefix = parsed.path.removeprefix("/")
    if not key_prefix:
        raise ValueError("s3_prefix must include a non-empty key prefix")
    return parsed.netloc, key_prefix
