# All-32 training handoff

Last updated: 2026-08-10

## Current state

- Branch: `codex/all32-shared-policy` (remote is up to date at `9bcec6d`).
- No MarioAI-All32 EC2 instance or launch reservation is active. Final
  reconciliation completed successfully.
- Training is still one shared `shared_complex_impala` policy. No specialist
  models have been introduced.
- All-32 acceptance has **not** passed. The README/GIF gallery still describe
  the legacy eight-level result; the all-32 acceptance report and final GIF
  refresh remain outstanding.

## Durable AWS artifacts

S3 root:

```text
s3://defectlens-phase3-002559670021/marioai/all32/
```

Phase 1 canonical head:

```text
models/all32-phase_1/latest.json
```

Its promoted checkpoint is `9,899,584` steps. The newer `10,649,536` candidate
was diagnosed but did not improve the promoted result (6 diagnostic clears,
3 levels with at least one clear). A later partial candidate at `10,944,512`
steps is retained as the undiagnosed `last_candidate`.

Phase 2 canonical head:

```text
models/all32-phase_2/latest.json
```

The promoted Phase-2 checkpoint is `11,899,584` steps, bootstrapped from the
promoted Phase-1 checkpoint. Its 3-rollout diagnostic produced 8 clears across
32 levels; 5 levels had at least one clear. The run ended early on a Spot
instance after about 0.96 billed hours. AWS did not retain a definitive
termination reason; the Spot interruption is the leading explanation. The
controller reconciled conservatively, so the settled ledger accounts for the
reserved four-hour run plus grace.

## Accounting snapshot

- Settled total: `$23.987404232869978...` of the `$50.00` cap.
- Phase 1: `$15.583039045743682...` of `$16.00`.
- Phase 2: `$2.900444444444444...` of `$16.00` (plus the shared launch-grace
  accounting line in the global total).
- Remaining global cap: `$26.012595767130021...`.

The local generated ledger is `reports/aws-spend.json`. The `.resume/` files
are controller staging/authoritative-ledger artifacts; preserve them, but do
not stage them casually with `git add -A`.

## Resume command

Resume Phase 2 from its canonical head, using the same bounded lifecycle:

```bash
.venv/bin/python scripts/aws_all32.py resume \
  --config configs/aws-all32.yaml \
  --ledger reports/aws-spend.json \
  --phase phase_2 \
  --max-hours 4.00 \
  --instance-type c7i.4xlarge \
  --checkpoint-s3-uri \
    s3://defectlens-phase3-002559670021/marioai/all32/models/all32-phase_2/latest.json \
  --ssh-key "$HOME/.ssh/mario-training-key.pem"
```

Run `status` and `reconcile` before any new paid action. The next acceptance
step is an official 15-stochastic-rollout evaluation for every one of the 32
levels. Only after that passes should the normalized 2x, infinite-loop GIFs
and README results be regenerated.
