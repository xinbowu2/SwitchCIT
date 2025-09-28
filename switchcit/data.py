from typing import Tuple
from datasets import load_dataset
import random
_PROMPT_PREFIX = (
    "Below is an instruction that describes a task, paired with an input that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
)
def _pack(instr: str, content: str, target: str) -> Tuple[str, str]:
    s = f"{_PROMPT_PREFIX}{instr}\n\n### Input:\n{content}\n\n### Response:"
    return s, target
def yield_simp(split="train", max_records=None, seed=0):
    random.seed(seed)
    if split == "train":
        ds = load_dataset("wiki_auto", "manual", split="train", trust_remote_code=True)
        recs = []
        for row in ds:
            src = row.get("source", "") or row.get("complex", "")
            tgt = row.get("target", "") or row.get("simple", "")
            if not src or not tgt: continue
            recs.append(_pack("Simplify the following sentence while preserving meaning.", src, tgt))
    else:
        ds = load_dataset("asset", split="validation", trust_remote_code=True)
        recs = []
        for row in ds:
            src = row.get("source", "")
            tgts = row.get("references", []) or row.get("target", [])
            if not src or not tgts: continue
            recs.append(_pack("Simplify the following sentence while preserving meaning.", src, tgts[0]))
    random.shuffle(recs)
    if max_records is not None: recs = recs[:max_records]
    for ex in recs: yield ex
def yield_emdg(split="train", max_records=None, seed=0):
    random.seed(seed)
    ds = load_dataset("empathetic_dialogues", split=split, trust_remote_code=True)
    recs = []
    for row in ds:
        ctx = row.get("context",""); utt = row.get("utterance",""); situation = row.get("situation","")
        if not ctx or not utt: continue
        content = f"Context: {ctx}\nSituation: {situation}"
        recs.append(_pack("Produce an empathetic response that acknowledges the emotion.", content, utt))
    random.shuffle(recs)
    if max_records is not None: recs = recs[:max_records]
    for ex in recs: yield ex
def yield_inq(split="train", max_records=None, seed=0):
    random.seed(seed)
    ds = load_dataset("eli5", split="train_eli5" if split=="train" else "validation_eli5", trust_remote_code=True)
    recs = []
    for row in ds:
        q = row.get("question",""); answers = row.get("answers",{}).get("text", []) if isinstance(row.get("answers"), dict) else row.get("answers", [])
        if not q or not answers: continue
        recs.append(_pack("Answer the question with a long-form explanation.", q, answers[0]))
    random.shuffle(recs)
    if max_records is not None: recs = recs[:max_records]
    for ex in recs: yield ex
def yield_exp(split="train", max_records=None, seed=0):
    random.seed(seed)
    ds = load_dataset("esnli", split=split, trust_remote_code=True)
    recs = []
    for row in ds:
        premise=row.get("premise",""); hypothesis=row.get("hypothesis","")
        exp = row.get("explanation_1","") or row.get("explanation_2","") or row.get("explanation_3","")
        if not premise or not hypothesis or not exp: continue
        content = f"Premise: {premise}\nHypothesis: {hypothesis}"
        recs.append(_pack("Explain the relationship between the premise and the hypothesis.", content, exp))
    random.shuffle(recs)
    if max_records is not None: recs = recs[:max_records]
    for ex in recs: yield ex
def yield_hgen(split="train", max_records=None, seed=0):
    random.seed(seed)
    ds = load_dataset("gigaword", split="train" if split=="train" else "validation", trust_remote_code=True)
    recs = []
    for row in ds:
        doc = row.get("document","") or row.get("source","") or row.get("text",""); summ = row.get("summary","") or row.get("headline","")
        if not doc or not summ: continue
        recs.append(_pack("Write a concise news headline for the text.", doc, summ))
    random.shuffle(recs)
    if max_records is not None: recs = recs[:max_records]
    for ex in recs: yield ex
YIELDERS = {"simp": yield_simp, "emdg": yield_emdg, "inq": yield_inq, "exp": yield_exp, "hgen": yield_hgen}
