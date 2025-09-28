import os, argparse, torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from switchcit.config import TASK_ORDER, DEFAULT_FEATURE_LLM
from switchcit import data as data_mod
from switchcit.switchnet import DenseClassificationModel, encode_instructions
from switchcit.metrics import rouge1, bleu, bert_score_f1, sari_proxy
from switchcit.config_utils import load_config

def load_expert(base_model_name: str, lora_dir: str):
    base = AutoModelForCausalLM.from_pretrained(base_model_name).to("cuda")
    return PeftModel.from_pretrained(base, lora_dir).to("cuda")

@torch.no_grad()
def infer_generate(model, tok, prompts, max_new_tokens=100):
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=tok.model_max_length).to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    texts = tok.batch_decode(out, skip_special_tokens=True)
    res = []
    for prompt, full in zip(prompts, texts):
        idx = full.find(prompt)
        res.append(full[idx+len(prompt):].strip() if idx >= 0 else full.strip())
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--base_model", type=str, default=None)
    ap.add_argument("--experts_dir", type=str, default="outputs/experts")
    ap.add_argument("--switchnet_dir", type=str, default=None)
    ap.add_argument("--feature_model", type=str, default=None)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--tasks", type=str, default=None)
    args = ap.parse_args()

    defaults = {"base_model": "bigscience/bloomz-1b1", "feature_model": DEFAULT_FEATURE_LLM, "tasks": TASK_ORDER,
                "eval": {"per_device_eval_batch_size": 8, "max_new_tokens": 100}}
    cfg = load_config(args.config, defaults)
    if args.base_model: cfg["base_model"] = args.base_model
    if args.feature_model: cfg["feature_model"] = args.feature_model
    if args.per_device_eval_batch_size: cfg["eval"]["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    if args.max_new_tokens: cfg["eval"]["max_new_tokens"] = args.max_new_tokens
    tasks = [t.strip() for t in (args.tasks.split(",") if args.tasks else cfg["tasks"]) if t.strip()]

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    switchnet = None
    if args.switchnet_dir and os.path.exists(os.path.join(args.switchnet_dir, "switchnet.pt")):
        switchnet = DenseClassificationModel(input_size=768, num_classes=len(tasks))
        switchnet.load_state_dict(torch.load(os.path.join(args.switchnet_dir, "switchnet.pt"), map_mode="cpu") if False else torch.load(os.path.join(args.switchnet_dir, "switchnet.pt"), map_location="cpu"))
        switchnet = switchnet.cuda().eval()
        print("[info] Using SwitchNet for routing.")
    else:
        print("[info] SwitchNet not found; evaluating direct experts.")

    results = {}
    for t in tasks:
        rows = list(data_mod.YIELDERS[t](split="validation", max_records=200))
        prompts = [s for s,_ in rows]; refs = [r for _,r in rows]
        expert_dir = os.path.join(args.experts_dir, t)
        if not os.path.exists(expert_dir):
            print(f"[warn] Expert {t} missing at {expert_dir}; skip."); continue
        preds = []
        bs = cfg["eval"]["per_device_eval_batch_size"]
        expert_cache = {}
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i+bs]
            if switchnet is None:
                if t not in expert_cache:
                    expert_cache[t] = load_expert(cfg["base_model"], expert_dir)
                preds.extend(infer_generate(expert_cache[t], tok, batch, cfg["eval"]["max_new_tokens"]))
            else:
                feats = encode_instructions(batch, cfg["feature_model"], device="cuda")
                logits = switchnet(feats.cuda())
                idxs = torch.argmax(logits, dim=-1).tolist()
                for p, did in zip(batch, idxs):
                    pred_task = tasks[did]
                    lora_path = os.path.join(args.experts_dir, pred_task)
                    if pred_task not in expert_cache:
                        expert_cache[pred_task] = load_expert(cfg["base_model"], lora_path)
                    preds.extend(infer_generate(expert_cache[pred_task], tok, [p], cfg["eval"]["max_new_tokens"]))
        if t == "simp":
            results[t] = {"SARI_proxy_ROUGE1": sari_proxy(preds, refs)}
        elif t == "hgen":
            results[t] = {"ROUGE1": rouge1(preds, refs), "BLEU": bleu(preds, refs)}
        else:
            results[t] = {"BERTScore_F1": bert_score_f1(preds, refs)}
        print(f"[{t}] {results[t]}")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics_incremental.json", "w") as f: json.dump(results, f, indent=2)
    print("[done] outputs/metrics_incremental.json")

if __name__ == "__main__":
    main()
