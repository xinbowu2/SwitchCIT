# SwitchCIT: Continual Instruction Tuning via Increasing Experts

Implementation of **SwitchCIT** — a continual instruction tuning framework that
adds parameter‑efficient **experts** as new tasks arrive, and learns a lightweight **Switch Network**
to route computations to the right expert at inference time.


## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```


## Configuration

Edit `configs/example.yaml` (or pass your own via `--config`).

```yaml
base_model: bigscience/bloomz-1b1
feature_model: facebook/opt-125m
tasks: [simp, emdg, inq, exp, hgen]

lora:
  r: 64
  alpha: 16
  dropout: 0.05
  bias: "none"

train:
  epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  max_records: 100000
  deepspeed: configs/deepspeed/zero2.json    # deepspeed configuration

switchnet:
  retention: 0.0001     # sample fraction from each task when training the switchnet
  epochs: 20
  lr: 0.001
  batch_size: 32

eval:
  per_device_eval_batch_size: 8
  max_new_tokens: 1024
```



## Quick Start

### Train one task

```bash
python scripts/train_experts.py   --config configs/example.yaml   --task simp   --eval_incremental
```

### Train all tasks 

```bash
python scripts/train_experts.py   --config configs/example.yaml   --eval_incremental
```

### Train Switch Network

```bash
python scripts/train_switchnet.py   --config configs/example.yaml
```

### Evaluate

```bash
# Direct experts (no routing)
python scripts/eval_switchcit.py --config configs/example.yaml --tasks simp,emdg

# Routed via SwitchNet (if available)
python scripts/eval_switchcit.py --config configs/example.yaml --tasks simp,emdg --switchnet_dir outputs/switchnet
```


## License

Apache-2.0



## Contributing

Pull requests welcome! For major changes, please open an issue to discuss what you would like to change.
