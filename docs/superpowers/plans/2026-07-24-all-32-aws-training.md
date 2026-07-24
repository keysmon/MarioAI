# All-32 Guarded AWS Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run staged all-32 shared-policy training on AWS Spot compute with durable checkpoints, auditable cost accounting, and automatic termination below the USD $50 hard cap.

**Architecture:** Pure budget logic is separated from an AWS CLI adapter so it can be fully unit-tested. A local orchestrator launches one tagged Spot instance using existing account resources, uploads the committed repository, starts a remote supervisor with its own runtime deadline, synchronizes checkpoints to an authorized S3 prefix, and terminates the instance. Phase promotion is driven only by immutable evaluation reports.

**Tech Stack:** Python 3.13 standard library, AWS CLI v2, EC2 Spot, S3, SSH/rsync, pytest

## Global Constraints

- AWS profile: `defectlens`; account: `002559670021`; region: `us-east-1`.
- Total phase budget is a hard USD $50 cap, including EC2 and attached gp3 time.
- Use one shared policy; no specialist training is authorized.
- Start with CPU-optimized Spot instances because NES environment stepping is CPU-bound.
- Persist checkpoints before interruption and terminate every paid instance after a run.
- A launch must be refused when its guarded maximum runtime does not fit the remaining budget.
- Training must stop and request a decision if shared-policy recovery exhausts its allocation.

---

## File Structure

- `src/marioai/budget.py`: ledger schema, cost accrual, allocation, and launch guard.
- `src/marioai/aws.py`: typed AWS CLI adapter and EC2/S3 parsing.
- `scripts/aws_all32.py`: preflight, launch, upload, monitor, sync, resume, and terminate CLI.
- `scripts/train_phase.py`: chunked training, diagnostic evaluation, best-checkpoint promotion, and regression reweighting.
- `scripts/cloud_train.sh`: instance-side timeout, training, sync, and shutdown supervisor.
- `configs/aws-all32.yaml`: exact account resources, instance candidates, allocations, and safety margins.
- `tests/test_budget.py`: pure cost and hard-cap tests.
- `tests/test_aws.py`: fake-command AWS orchestration tests.
- `scripts/aws_provision.md`: operator runbook generated from the implemented workflow.
- `reports/aws-spend.json`: runtime ledger created during execution and intentionally tracked with final evidence.

### Task 1: Durable budget ledger and hard-cap arithmetic

**Files:**
- Create: `src/marioai/budget.py`
- Create: `tests/test_budget.py`

**Interfaces:**
- Produces: `BudgetLedger.load(path: Path, cap_usd: Decimal = Decimal("50.00"))`
- Produces: `BudgetLedger.update_run(run: CostedRun) -> BudgetLedger`
- Produces: `BudgetLedger.remaining_usd: Decimal`
- Produces: `BudgetLedger.require_launch(phase: str, hourly_usd: Decimal, volume_hourly_usd: Decimal, max_hours: Decimal, reserve_usd: Decimal) -> None`
- Produces: atomic `BudgetLedger.save(path: Path)`.

- [ ] **Step 1: Write failing budget tests**

```python
from decimal import Decimal
import pytest
from marioai.budget import BudgetExceeded, BudgetLedger, CostedRun


def test_accrual_uses_decimal_and_includes_volume_time():
    ledger = BudgetLedger(cap_usd=Decimal("50.00"))
    updated = ledger.update_run(CostedRun(
        phase="benchmark",
        instance_id="i-test",
        hours=Decimal("2.5"),
        instance_hourly_usd=Decimal("0.60"),
        volume_hourly_usd=Decimal("0.011"),
    ))
    assert updated.spent_usd == Decimal("1.5275")


def test_launch_refused_before_projected_total_crosses_cap():
    ledger = BudgetLedger(
        cap_usd=Decimal("50.00"), spent_usd=Decimal("47.00")
    )
    with pytest.raises(BudgetExceeded, match="hard cap"):
        ledger.require_launch(
            "evaluation", Decimal("0.60"), Decimal("0.011"),
            Decimal("4"), Decimal("1.00")
        )


def test_atomic_roundtrip_preserves_runs(tmp_path):
    path = tmp_path / "spend.json"
    ledger.update_run(run).save(path)
    assert BudgetLedger.load(path) == ledger.update_run(run)


def test_progress_update_is_idempotent_and_monotonic():
    first = ledger.update_run(replace(run, hours=Decimal("1.0")))
    second = first.update_run(replace(run, hours=Decimal("1.5")))
    assert len(second.runs) == 1
    assert second.runs[0].hours == Decimal("1.5")
    with pytest.raises(ValueError, match="cannot decrease"):
        second.update_run(replace(run, hours=Decimal("1.0")))
```

- [ ] **Step 2: Verify budget tests fail**

Run: `.venv/bin/pytest tests/test_budget.py -v`

Expected: FAIL because `marioai.budget` does not exist.

- [ ] **Step 3: Implement immutable Decimal-based accounting**

Store decimal values as JSON strings. `CostedRun.cost_usd` equals
`hours * (instance_hourly_usd + volume_hourly_usd)`. Refuse negative durations,
decreasing updates for an existing instance/run pair, phase totals above their
allocation, or a projected `spent + run + reserve > cap`. Updating an existing
run replaces its prior accrued duration, making minute-by-minute persistence
idempotent instead of double-counting.

Use `tempfile.NamedTemporaryFile(dir=path.parent, delete=False)` followed by
`os.replace` for atomic writes.

- [ ] **Step 4: Run budget tests**

Run: `.venv/bin/pytest tests/test_budget.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marioai/budget.py tests/test_budget.py
git commit -m "feat: enforce durable AWS training budget"
```

### Task 2: Exact AWS configuration and read-only preflight

**Files:**
- Create: `configs/aws-all32.yaml`
- Create: `src/marioai/aws.py`
- Create: `tests/test_aws.py`

**Interfaces:**
- Produces: `AwsConfig.from_yaml(path: Path) -> AwsConfig`
- Produces: `AwsCli.run(args: Sequence[str]) -> dict | list | str`
- Produces: `AwsCli.preflight(config: AwsConfig) -> PreflightResult`
- Produces: `AwsCli.latest_spot_prices(instance_types: Sequence[str]) -> tuple[SpotOffer, ...]`

- [ ] **Step 1: Write failing configuration and preflight tests**

```python
def test_account_configuration_is_exact():
    cfg = AwsConfig.from_yaml(Path("configs/aws-all32.yaml"))
    assert cfg.profile == "defectlens"
    assert cfg.account_id == "002559670021"
    assert cfg.region == "us-east-1"
    assert cfg.security_group_id == "sg-03fc64395e32dea85"
    assert cfg.key_name == "mario-training-key"
    assert cfg.instance_profile == "defectlens-gpu-role"
    assert cfg.s3_prefix == "s3://defectlens-phase3-002559670021/marioai/all32/"
    assert cfg.cap_usd == Decimal("50.00")


def test_preflight_rejects_wrong_account(fake_aws):
    fake_aws.add(["sts", "get-caller-identity"], {"Account": "999999999999"})
    with pytest.raises(AwsPreflightError, match="expected AWS account"):
        fake_aws.preflight(config)


def test_spot_selection_uses_cheapest_allowed_offer(fake_aws):
    fake_aws.add_spot("c7i.8xlarge", "us-east-1f", "0.5568")
    fake_aws.add_spot("c7i.16xlarge", "us-east-1c", "0.8762")
    assert fake_aws.latest_spot_prices(config.instance_types)[0].instance_type == "c7i.8xlarge"
```

- [ ] **Step 2: Verify AWS tests fail**

Run: `.venv/bin/pytest tests/test_aws.py -v`

Expected: FAIL because the AWS adapter and config do not exist.

- [ ] **Step 3: Add exact account configuration**

```yaml
profile: defectlens
account_id: "002559670021"
region: us-east-1
ami_ssm_parameter: /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id
vpc_id: vpc-0ce3c6e06be6377df
subnet_ids:
  - subnet-0ba0242531d6615f3
  - subnet-0870eed3fd2b7dfab
  - subnet-02e9c5f8deb69ad62
  - subnet-0e5d82120140b0b4e
  - subnet-02ab8427ae66e6e36
  - subnet-0fa9ec503414b2be9
security_group_id: sg-03fc64395e32dea85
key_name: mario-training-key
instance_profile: defectlens-gpu-role
s3_prefix: s3://defectlens-phase3-002559670021/marioai/all32/
instance_types: [c7i.8xlarge, c7i.16xlarge]
on_demand_ceiling_usd:
  c7i.8xlarge: "1.428"
  c7i.16xlarge: "2.856"
root_volume_gb: 100
gp3_monthly_usd_per_gb: "0.08"
cap_usd: "50.00"
shutdown_threshold_usd: "49.00"
grace_minutes: 15
allocations:
  benchmark: "4.00"
  phase_1: "16.00"
  phase_2: "16.00"
  recovery: "10.00"
  evaluation: "4.00"
```

- [ ] **Step 4: Implement safe subprocess parsing**

Always invoke AWS as an argument vector:

```python
command = [
    "aws", *args, "--profile", self.profile,
    "--region", self.region, "--output", "json",
]
completed = subprocess.run(
    command, check=True, text=True, capture_output=True, timeout=60
)
```

Preflight verifies caller account, VPC, all subnets, security group, key pair,
instance profile, S3 list access, and that no running instance tagged
`Project=MarioAI-All32` exists. It must be read-only.

- [ ] **Step 5: Run AWS adapter tests**

Run: `.venv/bin/pytest tests/test_aws.py -v`

Expected: PASS with no live AWS calls.

- [ ] **Step 6: Commit**

```bash
git add configs/aws-all32.yaml src/marioai/aws.py tests/test_aws.py
git commit -m "feat: validate exact AWS all-32 prerequisites"
```

### Task 3: Spot launch, dual shutdown guards, and termination

**Files:**
- Create: `scripts/aws_all32.py`
- Create: `scripts/cloud_train.sh`
- Modify: `tests/test_aws.py`

**Interfaces:**
- Consumes: `AwsConfig`, `AwsCli`, `BudgetLedger`.
- Produces: `launch_guarded_instance(phase: str, max_hours: Decimal) -> LaunchedInstance`
- Produces: `monitor_and_terminate(instance: LaunchedInstance, ledger_path: Path) -> CostedRun`
- Produces CLI subcommands `preflight`, `launch`, `status`, `terminate`, and `reconcile`.

- [ ] **Step 1: Write failing launch lifecycle tests**

```python
def test_launch_sets_one_time_spot_and_terminate_on_shutdown(orchestrator):
    instance = orchestrator.launch_guarded_instance(
        phase="benchmark", max_hours=Decimal("1.0")
    )
    request = orchestrator.aws.last_run_instances_request
    assert request["InstanceMarketOptions"]["SpotOptions"]["SpotInstanceType"] == "one-time"
    assert request["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert request["TagSpecifications"][0]["Tags"] == [
        {"Key": "Project", "Value": "MarioAI-All32"},
        {"Key": "Phase", "Value": "benchmark"},
    ]


def test_launch_checks_budget_before_run_instances(orchestrator):
    orchestrator.ledger = nearly_exhausted_ledger()
    with pytest.raises(BudgetExceeded):
        orchestrator.launch_guarded_instance("phase_2", Decimal("2"))
    assert not orchestrator.aws.run_instances_called


def test_monitor_terminates_after_process_exit(orchestrator):
    orchestrator.remote.exit_code = 1
    orchestrator.monitor_and_terminate(instance, ledger_path)
    assert orchestrator.aws.terminated_ids == [instance.instance_id]
```

- [ ] **Step 2: Verify lifecycle tests fail**

Run: `.venv/bin/pytest tests/test_aws.py -v`

Expected: FAIL because the orchestrator does not exist.

- [ ] **Step 3: Implement guarded launch**

Resolve the current AMI from the configured SSM public parameter and choose the
cheapest available allowed instance/AZ pair. Accrue observed cost at the live
Spot rate, but compute the maximum permitted runtime using the configured
Northern Virginia on-demand ceiling (`$1.428` for `c7i.8xlarge`, `$2.856` for
`c7i.16xlarge`) plus gp3 hourly cost. This conservative ceiling keeps the hard
cap valid even if Spot pricing rises after launch. Call
`BudgetLedger.require_launch` before `run-instances`.

Launch with a 100 GB encrypted gp3 root volume that deletes on termination,
public IP, the existing security group/key/profile, one-time Spot market
options, and instance-initiated shutdown behavior `terminate`.

- [ ] **Step 4: Implement the independent remote supervisor**

`cloud_train.sh` accepts phase, maximum seconds, repository directory, and S3
prefix. It must install an EXIT trap before training:

```bash
finish() {
  aws s3 sync "$REPO_DIR/models/" "${S3_PREFIX}models/"
  aws s3 sync "$REPO_DIR/reports/" "${S3_PREFIX}reports/"
  sudo shutdown -h now
}
trap finish EXIT INT TERM
timeout --signal=TERM --kill-after=300 "$MAX_SECONDS" \
  .venv/bin/python -m marioai.train "$@"
```

The local monitor polls at most once per minute, persists elapsed cost after
each successful poll, requests remote termination at the safety threshold, and
calls EC2 termination in a `finally` block even when SSH, rsync, or training
fails.

- [ ] **Step 5: Run lifecycle tests**

Run: `.venv/bin/pytest tests/test_aws.py tests/test_budget.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/aws_all32.py scripts/cloud_train.sh tests/test_aws.py
git commit -m "feat: launch and terminate budget-guarded Spot training"
```

### Task 4: Checkpoint upload, resume, and interruption recovery

**Files:**
- Modify: `src/marioai/train.py`
- Modify: `scripts/aws_all32.py`
- Modify: `scripts/cloud_train.sh`
- Modify: `tests/test_train.py`
- Modify: `tests/test_aws.py`

**Interfaces:**
- Consumes: training `--resume` and immutable run configuration.
- Produces CLI subcommand `resume --phase ... --checkpoint-s3-uri ...`.
- Produces: checkpoint bundles containing model ZIP, run config, timestep, and budget-ledger snapshot.

- [ ] **Step 1: Write failing interruption tests**

```python
def test_checkpoint_callback_writes_resume_manifest(tmp_path):
    callback = DurableCheckpointCallback(save_path=tmp_path, save_freq=1)
    callback.save_checkpoint(fake_model(num_timesteps=250000))
    manifest = json.loads((tmp_path / "latest.json").read_text())
    assert manifest["num_timesteps"] == 250000
    assert manifest["model"] == "ckpt_250000_steps.zip"


def test_resume_downloads_exact_manifest_checkpoint(orchestrator):
    orchestrator.aws.s3_object("runs/phase_1/latest.json", manifest)
    path = orchestrator.restore_latest("phase_1")
    assert path.name == manifest["model"]
    assert orchestrator.verified_sha256(path) == manifest["sha256"]
```

- [ ] **Step 2: Verify interruption tests fail**

Run: `.venv/bin/pytest tests/test_train.py tests/test_aws.py -v`

Expected: FAIL because durable manifests are not written.

- [ ] **Step 3: Add checkpoint manifests and sync cadence**

Subclass `CheckpointCallback` so every save atomically updates `latest.json`
with timestep, filename, SHA-256, action set, extractor class, and phase.
`cloud_train.sh` runs a background S3 sync every 15 minutes and performs a final
sync in its trap.

- [ ] **Step 4: Implement verified restore**

The restore path downloads `latest.json`, downloads only the named checkpoint
and run config, verifies SHA-256, then invokes `train --resume`. Reject a phase,
action set, extractor, or checkpoint hash mismatch before launching paid
compute.

- [ ] **Step 5: Run checkpoint and orchestration tests**

Run: `.venv/bin/pytest tests/test_train.py tests/test_aws.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/marioai/train.py scripts/aws_all32.py scripts/cloud_train.sh tests/test_train.py tests/test_aws.py
git commit -m "feat: resume all-32 training from durable checkpoints"
```

### Task 5: Cloud runbook and measured benchmark gate

**Files:**
- Rewrite: `scripts/aws_provision.md`
- Modify: `scripts/aws_all32.py`
- Create: `scripts/train_phase.py`
- Modify: `tests/test_aws.py`

**Interfaces:**
- Produces CLI subcommand `benchmark --max-spend 4.00`.
- Produces benchmark report fields `env_steps_per_second`, `cost_per_million_steps`, `peak_rss_gb`, and selected instance type.
- Produces: `run_phase(phase: str, deadline: datetime, checkpoint: Path | None) -> Path`.

- [ ] **Step 1: Write failing benchmark-selection test**

```python
def test_benchmark_selects_lower_cost_per_million_steps():
    offers = [
        Benchmark("c7i.8xlarge", 1800.0, Decimal("0.5568")),
        Benchmark("c7i.16xlarge", 2900.0, Decimal("0.9646")),
    ]
    assert select_benchmark(offers).instance_type == "c7i.8xlarge"


def test_phase_loop_promotes_only_better_coverage_and_reweights_regressions(fake_phase):
    fake_phase.reports = [
        report(passing={"1-1"}),
        report(passing={"1-1", "1-2"}),
        report(passing={"1-2"}),
    ]
    best = fake_phase.run()
    assert best.report.passing_levels == {"1-1", "1-2"}
    assert fake_phase.next_weights["1-1"] == 2.0
```

- [ ] **Step 2: Verify benchmark test fails**

Run: `.venv/bin/pytest tests/test_aws.py::test_benchmark_selects_lower_cost_per_million_steps -v`

Expected: FAIL because benchmark selection is absent.

- [ ] **Step 3: Implement benchmark report and runbook**

Run 250,000 environment steps on each candidate only while the benchmark
allocation remains. Compute:

```python
cost_per_million = (
    (instance_hourly_usd + volume_hourly_usd)
    * Decimal("1000000")
    / Decimal(str(env_steps_per_second))
    / Decimal("3600")
)
```

The runbook must contain exact preflight, benchmark, launch, status, reconcile,
and emergency terminate commands using `configs/aws-all32.yaml`. It must state
that a non-empty running-instance query blocks another launch.

`train_phase.py` trains `chunk_timesteps` at a time, runs three seeded
stochastic diagnostic rollouts per active stage, promotes a checkpoint only
when `is_better_checkpoint` improves coverage/clears/progress, computes
`regression_weights` for the next fixed-worker assignment, and resumes until
the configured total timesteps or the remote deadline. Diagnostic reports are
stored separately and cannot satisfy the 15-rollout acceptance predicate.

- [ ] **Step 4: Run AWS tests and a live read-only preflight**

Run: `.venv/bin/pytest tests/test_aws.py tests/test_budget.py -v`

Expected: PASS.

Run: `.venv/bin/python scripts/aws_all32.py preflight --config configs/aws-all32.yaml`

Expected: account `002559670021`, no running `MarioAI-All32` instance, accessible
S3 prefix, and at least one valid Spot offer. This command launches nothing.

- [ ] **Step 5: Commit**

```bash
git add scripts/aws_provision.md scripts/aws_all32.py scripts/train_phase.py tests/test_aws.py
git commit -m "docs: add guarded all-32 AWS runbook"
```

### Task 6: Execute staged shared-policy training

**Files:**
- Create during execution: `reports/aws-spend.json`
- Create during execution: `reports/phase-1.json`
- Create during execution: `reports/phase-2.json`
- Create during execution: `reports/final-acceptance.json`

**Interfaces:**
- Consumes: completed local pipeline, guarded AWS runner, and committed code.
- Produces: best shared checkpoint plus evaluation and spend evidence.

- [ ] **Step 1: Push the tested implementation before paid execution**

Run: `git status --short`

Expected: empty.

Run: `git push origin main`

Expected: the exact tested commit is available to the training instance.

- [ ] **Step 2: Run the benchmark allocation**

Run: `.venv/bin/python scripts/aws_all32.py benchmark --config configs/aws-all32.yaml --ledger reports/aws-spend.json --max-spend 4.00`

Expected: both candidate results or an early statistically decisive result,
the selected cost-per-million winner, a terminated instance, and cumulative
spend no more than $4.

- [ ] **Step 3: Train and evaluate worlds 1-4**

Run: `.venv/bin/python scripts/aws_all32.py launch --config configs/aws-all32.yaml --ledger reports/aws-spend.json --phase phase_1 --allocation 16.00`

After artifact sync, run:

```bash
.venv/bin/python -m marioai.evaluate \
  --model models/all32-phase1/best.zip \
  --levels 1-1 1-2 1-3 1-4 2-1 2-2 2-3 2-4 3-1 3-2 3-3 3-4 4-1 4-2 4-3 4-4 \
  --episodes 15 --stochastic --seed 42000 --out reports/phase-1.json
```

Promote only if the report contains the same checkpoint hash and improved stage
coverage. Terminate and reconcile before the next launch.

- [ ] **Step 4: Train and evaluate all 32**

Run: `.venv/bin/python scripts/aws_all32.py launch --config configs/aws-all32.yaml --ledger reports/aws-spend.json --phase phase_2 --allocation 16.00 --resume models/all32-phase1/best.zip`

Run:

```bash
.venv/bin/python -m marioai.evaluate \
  --model models/all32-phase2/best.zip --levels all \
  --episodes 15 --stochastic --seed 42000 \
  --out reports/phase-2.json
```

Expected: a truthful coverage count and per-stage failure evidence for the same
shared checkpoint.

- [ ] **Step 5: Apply shared-policy recovery only to measured failures**

For each failed stage, classify the trace. For 4-4 and 7-4, compare the stacked
pixel observations at branch decisions. If the same four-frame observation
requires different actions based on earlier route history, stop and write a
focused recurrent-policy design amendment before spending the recovery
allocation. Likewise, if identical observations across stages require
conflicting actions, write a focused stable level-conditioning amendment. Do
not add recurrence or conditioning based only on low reward or poor
exploration.

Generate aligned routes only for exploration failures, clone all generated
routes into the same checkpoint in one balanced batch, and spend at most the
recovery allocation:

```bash
.venv/bin/python scripts/behavior_clone_route.py \
  --route-dir models/routes/2-2 \
  --route-dir models/routes/4-4 \
  --action-set complex --skip 4 \
  --init-from models/all32-phase2/best.zip \
  --out models/all32-recovery/bc.zip
```

The actual repeated `--route-dir` arguments must be exactly the failure set
recorded in `reports/phase-2.json`; do not create specialists.

- [ ] **Step 6: Run final acceptance and stop**

Run:

```bash
.venv/bin/python -m marioai.evaluate \
  --model models/all32-final/best.zip --levels all \
  --episodes 15 --stochastic --seed 42000 \
  --out reports/final-acceptance.json
.venv/bin/python scripts/aws_all32.py reconcile \
  --config configs/aws-all32.yaml --ledger reports/aws-spend.json
```

Expected: all AWS instances terminated, spend at or below $50, and either:

- `final-acceptance.json` passes 32/32 for one checkpoint; or
- paid execution stops with the best shared checkpoint and a user-facing
  specialist decision report.

- [ ] **Step 7: Commit evidence**

```bash
git add reports/aws-spend.json reports/phase-1.json reports/phase-2.json reports/final-acceptance.json
git commit -m "results: record all-32 shared-policy evaluation"
```
