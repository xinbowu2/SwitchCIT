from typing import List
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
class DenseClassificationModel(nn.Module):
    def __init__(self, input_size=768, num_classes=2):
        super().__init__()
        self.input = nn.Linear(input_size, 200)
        self.act = nn.ReLU()
        self.linear = nn.Linear(200, num_classes)
    def forward(self, x):
        return self.linear(self.act(self.input(x)))
@torch.no_grad()
def encode_instructions(texts: List[str], model_name: str, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModel.from_pretrained(model_name).to(device)
    feats = []
    for t in texts:
        inputs = tok(t, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = m(**inputs).last_hidden_state
        feats.append(out[:, -1, :].squeeze(0).detach().cpu())
    return torch.stack(feats, dim=0)
def save(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
def load(path: str, input_size=768, num_classes=2):
    model = DenseClassificationModel(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
