import os, argparse, random, torch
from torch.utils.data import Dataset
import torch.nn as nn
from switchcit.config import TASK_ORDER, DEFAULT_FEATURE_LLM
from switchcit import data as data_mod
from switchcit.switchnet import DenseClassificationModel, encode_instructions
from switchcit.config_utils import load_config

class SwitchDataset(Dataset):
    def __init__(self, tasks, retention: float = 0.0001, seed: int = 0):
        random.seed(seed)
        self.rows = []
        for t in tasks[1:]:
            exs = list(data_mod.YIELDERS[t](split="train", max_records=100000))
            k = max(1, int(len(exs) * retention))
            import random as _r
            for src, _ in _r.sample(exs, k):
                self.rows.append((t, src))
        random.shuffle(self.rows)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="outputs/switchnet")
    ap.add_argument("--feature_model", type=str, default=None)
    ap.add_argument("--tasks", type=str, default=None)
    args = ap.parse_args()

    defaults = {"feature_model": DEFAULT_FEATURE_LLM, "tasks": TASK_ORDER,
                "switchnet": {"retention": 0.0001, "epochs": 20, "lr": 1e-3, "batch_size": 32}}
    cfg = load_config(args.config, defaults)
    if args.feature_model: cfg["feature_model"] = args.feature_model
    tasks = [t.strip() for t in (args.tasks.split(",") if args.tasks else cfg["tasks"]) if t.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    ds = SwitchDataset(tasks, retention=cfg["switchnet"]["retention"])
    model = DenseClassificationModel(input_size=768, num_classes=len(tasks)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["switchnet"]["lr"])
    crit = nn.CrossEntropyLoss()

    for epoch in range(cfg["switchnet"]["epochs"]):
        random.shuffle(ds.rows)
        total, n = 0.0, 0
        for i in range(0, len(ds), cfg["switchnet"]["batch_size"]):
            batch = ds.rows[i:i+cfg["switchnet"]["batch_size"]]
            labels = [tasks.index(t) for t, _ in batch]
            texts = [s for _, s in batch]
            feats = encode_instructions(texts, cfg["feature_model"], device="cuda")
            logits = model(feats.cuda())
            loss = crit(logits, torch.tensor(labels, dtype=torch.long, device="cuda"))
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())*len(batch); n += len(batch)
        print(f"[epoch {epoch+1}/{cfg['switchnet']['epochs']}] loss={total/max(1,n):.4f}")
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"sn_epoch_{epoch+1}.pt"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, "switchnet.pt"))
    print("[done] Saved switchnet checkpoints.")

if __name__ == "__main__":
    main()
