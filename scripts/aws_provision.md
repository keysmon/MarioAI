# AWS GPU Training Runbook (defectlens, us-east-1, <$50, on-demand)

Real training runs on an AWS on-demand GPU. Local machines run only the Phase 0
spike and short sanity checks. Follow these steps in order.

## 0. Identity + quota preflight (do FIRST)

```bash
aws sts get-caller-identity --profile defectlens
# Confirm account 002559670021.

aws service-quotas get-service-quota --service-code ec2 \
  --quota-code L-DB2E81BA --region us-east-1 --profile defectlens
# L-DB2E81BA = "Running On-Demand G and VT instances" (measured in vCPUs).
# g5.2xlarge needs 8 vCPUs. If the current value is < 8, request an increase and WAIT
# (approval can take hours):
aws service-quotas request-service-quota-increase --service-code ec2 \
  --quota-code L-DB2E81BA --desired-value 8 --region us-east-1 --profile defectlens
```

## 1. Launch on-demand g5.2xlarge (Deep Learning AMI)

Use the `launching-ec2-instance-with-best-practices` skill, OR launch manually with a
recent Deep Learning OSS PyTorch AMI (Ubuntu, us-east-1), a 100 GB gp3 root volume, an
SSH key, and a security group allowing only your IP on port 22. Tag `Project=MarioAI`.

g5.2xlarge = 8 vCPUs + 1x A10G GPU. On-demand ~$1.21/hr; a full multi-task run is
roughly 1-4 GPU-hours, comfortably under the $50 ceiling.

## 2. Sync code + install (Linux CUDA torch)

```bash
# from your Mac:
rsync -av --exclude .venv --exclude models --exclude runs --exclude .git \
  ./ ubuntu@<ip>:~/MarioAI/

# on the instance:
ssh ubuntu@<ip>
cd MarioAI
python3.13 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Swap the CPU torch wheel for the CUDA build (matching the frozen torch version):
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e . --no-deps
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # expect: cuda: True
```

If the exact CUDA torch wheel is unavailable, install the nearest available CUDA build
of torch; the RL code is torch-version-tolerant. Re-run the spike as a sanity check:
`python scripts/spike.py`.

## 3. Train (see plan Tasks 10-11)

```bash
# Milestone 1 - 1-1 proof gate:
python -m marioai.train --config configs/default.yaml --levels 1-1 \
  --timesteps 1000000 --run-name mario_1_1
# Milestone 2 - multi-task:
python -m marioai.train --config configs/default.yaml --run-name mario_multitask
```

Watch progress (SSH-tunnel TensorBoard):

```bash
tensorboard --logdir runs --port 6006
# from your Mac: ssh -L 6006:localhost:6006 ubuntu@<ip>, then open localhost:6006
```

## 4. Record GIFs (headless), retrieve artifacts, then TERMINATE

`nes-py` renders `rgb_array` with no display, so GIF recording works on a bare box.

```bash
# on the instance:
scripts/record_all_gifs.sh models/mario_multitask/final.zip

# from your Mac - pull artifacts down:
rsync -av ubuntu@<ip>:~/MarioAI/models/ ./models/
rsync -av ubuntu@<ip>:~/MarioAI/assets/gifs/ ./assets/gifs/

# TERMINATE (not stop - a stopped instance still bills for its EBS volume):
aws ec2 terminate-instances --instance-ids <id> --region us-east-1 --profile defectlens

# Confirm nothing is still running:
aws ec2 describe-instances --region us-east-1 --profile defectlens \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].InstanceId'
```
