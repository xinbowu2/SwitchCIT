from typing import List
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore
def rouge1(preds: List[str], refs: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    total = 0.0
    for p, r in zip(preds, refs):
        total += scorer.score(r, p)["rouge1"].fmeasure
    return total / max(1, len(preds))
def bleu(preds: List[str], refs: List[str]) -> float:
    refs_list = [refs]
    return sacrebleu.corpus_bleu(preds, refs_list).score
def bert_score_f1(preds: List[str], refs: List[str]) -> float:
    P, R, F1 = bertscore(preds, refs, lang="en", rescale_with_baseline=True)
    return float(F1.mean())
def sari_proxy(preds: List[str], refs: List[str]) -> float:
    return rouge1(preds, refs)
