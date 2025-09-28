import os, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from switchcit.config import TASK_ORDER, LORA
from switchcit import data as data_mod
from switchcit.config_utils import load_config

class TaskDataset(Dataset):
    def __init__(self, task: str, split: str, tokenizer, max_records=None):
        self.task = task
        self.tokenizer = tokenizer
        self.rows = list(data_mod.YIELDERS[task](split=split, max_records=max_records))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        src, tgt = self.rows[i]
        ids = self.tokenizer(src + " " + tgt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length)
        input_ids = ids["input_ids"][0]
        labels = input_ids.clone()
        src_ids = self.tokenizer(src, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length)["input_ids"][0]
        labels[: len(src_ids)] = -100
        return {"input_ids": input_ids, "labels": labels}

def collate(batch, pad_id):
    import torch
    ids = [b["input_ids"] for b in batch]
    lbls = [b["labels"] for b in batch]
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
    lbls = torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True, padding_value=-100)
    attn = ids.ne(pad_id)
    return {"input_ids": ids, "labels": lbls, "attention_mask": attn}

def maybe_eval_incremental(base_model, experts_dir, switchnet_dir, tasks_to_eval, feature_model, per_device_eval_batch_size, max_new_tokens):
    import subprocess, sys, os
    eval_script = os.path.join(os.path.dirname(__file__), "eval_switchcit.py")
    cmd = [sys.executable, eval_script, "--base_model", base_model, "--experts_dir", experts_dir,
           "--per_device_eval_batch_size", str(per_device_eval_batch_size), "--max_new_tokens", str(max_new_tokens),
           "--tasks", ",".join(tasks_to_eval), "--feature_model", feature_model]
    if switchnet_dir and os.path.exists(os.path.join(switchnet_dir, "switchnet.pt")):
        cmd += ["--switchnet_dir", switchnet_dir]
    print(f"[incremental-eval] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--task", type=str, default=None, choices=TASK_ORDER)
    ap.add_argument("--output_dir", type=str, default="outputs/experts")
    ap.add_argument("--eval_incremental", action="store_true")
    ap.add_argument("--switchnet_dir", type=str, default="outputs/switchnet")
    ap.add_argument("--base_model", type=str, default=None)
    args = ap.parse_args()

    defaults = {
        "base_model": "bigscience/bloomz-1b1",
        "tasks": TASK_ORDER,
        "lora": {"r": LORA.r, "alpha": LORA.alpha, "dropout": LORA.dropout, "bias": LORA.bias},
        "train": {"epochs": 3, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
                  "learning_rate": 2e-5, "max_records": 100000, "deepspeed": None},
        "eval": {"per_device_eval_batch_size": 8, "max_new_tokens": 100},
        "feature_model": "facebook/opt-125m",
    }
    cfg = load_config(args.config, defaults)
    if args.base_model: cfg["base_model"] = args.base_model
    tasks = [args.task] if args.task else cfg["tasks"]

    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    for t in tasks:
        task_dir = os.path.join(args.output_dir, t)
        if os.path.exists(task_dir) and os.listdir(task_dir):
            print(f"[skip] {t} exists at {task_dir}")
            if args.eval_incremental:
                maybe_eval_incremental(cfg["base_model"], args.output_dir, args.switchnet_dir, [t],
                                       cfg["feature_model"], cfg["eval"]["per_device_eval_batch_size"], cfg["eval"]["max_new_tokens"])
            continue

        print(f"==> Training expert for task [{t}]")
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])
        lc = cfg["lora"]
        from peft import LoraConfig
        lora_cfg = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias=lc["bias"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)

        ds = TaskDataset(t, "train", tok, max_records=cfg["train"]["max_records"])
        coll = lambda b: collate(b, tok.pad_token_id)
        targs = TrainingArguments(
            output_dir=task_dir, num_train_epochs=cfg["train"]["epochs"],
            per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
            learning_rate=cfg["train"]["learning_rate"],
            optim="adamw_torch", lr_scheduler_type="constant",
            logging_steps=10, save_steps=10_000_000, save_strategy="steps", save_total_limit=1,
            deepspeed=cfg["train"]["deepspeed"], report_to=[]
        )
        trainer = Trainer(model=model, args=targs, data_collator=coll, train_dataset=ds, tokenizer=tok)
        trainer.train(); trainer.save_model(task_dir)
        print(f"[done] Saved expert LoRA to {task_dir}")

        if args.eval_incremental:
            maybe_eval_incremental(cfg["base_model"], args.output_dir, args.switchnet_dir, [t],
                                   cfg["feature_model"], cfg["eval"]["per_device_eval_batch_size"], cfg["eval"]["max_new_tokens"])

if __name__ == "__main__":
    main()
