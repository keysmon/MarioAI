# Guarded all-32 AWS runbook

This workflow is authorized only for AWS profile `defectlens`, account
`002559670021`, region `us-east-1`, and the resources pinned in
`configs/aws-all32.yaml`. It trains one shared complex-action IMPALA policy.
Do not add specialist policies, recurrence, or level conditioning during this
run.

Run every command from the repository root with a clean, tested commit. The
ledger is `reports/aws-spend.json`; never delete or roll it back. All lifecycle
commands use argument vectors, the configured on-demand ceilings for the
safety ledger, and unconditional instance termination. The benchmark report
uses the exact observed Spot rate and duration only for measured
cost-per-million selection.

## 1. Read-only preflight and launch blocker

```bash
.venv/bin/python scripts/aws_all32.py preflight \
  --config configs/aws-all32.yaml
```

The JSON must identify account `002559670021`, the configured VPC/subnets,
accessible S3 prefix, and at least one current allowed Spot offer. It must show
no running `MarioAI-All32` instance.

For an independent operator check, run the exact query used as the launch
blocker:

```bash
aws ec2 describe-instances \
  --profile defectlens \
  --region us-east-1 \
  --filters \
    Name=tag:Project,Values=MarioAI-All32 \
    Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
  --query 'Reservations[].Instances[].InstanceId' \
  --output json
```

Any non-empty result blocks another launch. Run `reconcile`; do not start a
second instance.

## 2. Measured benchmark gate — maximum USD 4.00

The fixed workload is exactly 250,000 Stable-Baselines environment/action
steps. With frame skip 4, these are decision steps, not one million emulator
frames.

```bash
mkdir -p reports
set -o pipefail
.venv/bin/python scripts/aws_all32.py benchmark \
  --config configs/aws-all32.yaml \
  --ledger reports/aws-spend.json \
  --max-spend 4.00 \
  --ssh-key "$HOME/.ssh/mario-training-key.pem" \
  | tee reports/aws-benchmark.json
```

The command measures both configured candidates while the allocation can fit
the next candidate's conservative runtime plus the single termination grace
window. Otherwise it returns the best completed measurement with
`decision: allocation_exhausted`. It always terminates and settles a launched
candidate. The report records each candidate's `env_steps_per_second`,
`cost_per_million_steps`, `peak_rss_gb`, and the
`selected_instance_type`.

Load the measured winner for later commands:

```bash
INSTANCE_TYPE="$(
  .venv/bin/python -c \
  'import json; print(json.load(open("reports/aws-benchmark.json", encoding="utf-8"))["selected_instance_type"])'
)"
printf '%s\n' "$INSTANCE_TYPE"
```

Only `c7i.8xlarge` or `c7i.16xlarge` is valid. The lifecycle command rejects
any other value before mutation.

## 3. Launch and status

Example phase-1 launch with a four-hour guarded maximum:

```bash
.venv/bin/python scripts/aws_all32.py launch \
  --config configs/aws-all32.yaml \
  --ledger reports/aws-spend.json \
  --phase phase_1 \
  --max-hours 4.00 \
  --instance-type "$INSTANCE_TYPE" \
  --ssh-key "$HOME/.ssh/mario-training-key.pem"
```

The launch performs another full preflight immediately before mutation,
persists a crash-recoverable reservation first, and refuses a maximum runtime
that cannot fit the phase allocation or hard USD 50 cap. Do not bypass the
command with a raw `run-instances` call.

Check all project instances, or one exact ID:

```bash
.venv/bin/python scripts/aws_all32.py status \
  --config configs/aws-all32.yaml

.venv/bin/python scripts/aws_all32.py status \
  --config configs/aws-all32.yaml \
  --instance-id i-0123456789abcdef0
```

## 4. Reconcile after every run

```bash
.venv/bin/python scripts/aws_all32.py reconcile \
  --config configs/aws-all32.yaml \
  --ledger reports/aws-spend.json
```

Reconciliation terminates active `Project=MarioAI-All32` resources, settles
the durable reservation conservatively, and clears it only after exact
terminal confirmation. Repeat `status` and the independent blocker query
until both are empty.

## 5. Emergency termination

First confirm the exact Project-tagged target with `status`, then request
idempotent termination:

```bash
.venv/bin/python scripts/aws_all32.py terminate \
  --config configs/aws-all32.yaml \
  --instance-id i-0123456789abcdef0

.venv/bin/python scripts/aws_all32.py reconcile \
  --config configs/aws-all32.yaml \
  --ledger reports/aws-spend.json
```

The command refuses an unowned or malformed instance ID. If SSH, training,
sync, or accounting fails, preserve the error and run `reconcile`; never erase
the launch sidecar manually.

## 6. Resume an exact durable bundle

Use only the manifested S3 object for the same shared phase:

```bash
.venv/bin/python scripts/aws_all32.py resume \
  --config configs/aws-all32.yaml \
  --ledger reports/aws-spend.json \
  --phase phase_1 \
  --max-hours 4.00 \
  --instance-type "$INSTANCE_TYPE" \
  --checkpoint-s3-uri \
    s3://defectlens-phase3-002559670021/marioai/all32/models/all32-phase_1/latest.json \
  --ssh-key "$HOME/.ssh/mario-training-key.pem"
```

Resume downloads only the manifest-named model/config/signature,
VecNormalize state, and ledger snapshot. Hash, policy/environment identity,
timestep, and authoritative-ledger checks all occur before paid mutation.
Chunk progress is expressed in environment steps; a resumed checkpoint trains
only the remaining phase target.

## 7. Spend and evidence checks

```bash
.venv/bin/python -c \
'from decimal import Decimal
from pathlib import Path
from marioai.budget import BudgetLedger
ledger = BudgetLedger.load(Path("reports/aws-spend.json"), cap_usd=Decimal("50.00"))
print("spent_usd =", ledger.spent_usd)
print("remaining_usd =", ledger.remaining_usd)
print("runs =", len(ledger.runs))
assert ledger.spent_usd <= Decimal("50.00")'
```

Before another paid action, also inspect the benchmark and diagnostic evidence:

```bash
.venv/bin/python -m json.tool reports/aws-benchmark.json
find reports/diagnostics -type f -name '*.json' -print | sort
```

Chunk diagnostics are permanently separate from acceptance evidence: exactly
three seeded stochastic rollouts per active stage can rank checkpoints through
coverage, clears, and progress, but can never satisfy the final 15-rollout
acceptance predicate. A regressed stage receives deterministic doubled weight
in the next fixed-worker shared-policy assignment.
