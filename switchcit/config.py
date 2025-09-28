from dataclasses import dataclass
TASK_ORDER = ["simp", "emdg", "inq", "exp", "hgen"]
DEFAULT_BASE_LLM = "bigscience/bloomz-1b1"
DEFAULT_FEATURE_LLM = "facebook/opt-125m"
@dataclass
class LoraCfg:
    r: int = 64
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
LORA = LoraCfg()
