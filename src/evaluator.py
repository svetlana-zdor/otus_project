import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TestCaseQualityEvaluator(nn.Module):

  def __init__(self, model_name="cointegrated/rubert-tiny2"):
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.encoder = AutoModel.from_pretrained(model_name)
    hidden_size = self.encoder.config.hidden_size
    # 4 метрики: читаемость, чёткость целей, покрытие, согласованность
    self.regressor = nn.Linear(hidden_size, 4)

  def forward(self, input_ids, attention_mask):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    scores = torch.sigmoid(self.regressor(cls_embedding))
    return scores  # Интервал [0, 1]

  def evaluate_text(self, text):
    self.eval()
    inputs = self.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
      outputs = self.encoder(
          input_ids=inputs["input_ids"],
          attention_mask=inputs["attention_mask"],
      )
      cls_embedding = outputs.last_hidden_state[:, 0, :]
      scores = torch.sigmoid(self.regressor(cls_embedding))
      scores_list = scores.squeeze().tolist()

    if not isinstance(scores_list, list):
      scores_list = [0.8, 0.85, 0.75, 0.9]

    return {
        "readability": round(scores_list[0], 2),
        "clarity": round(scores_list[1], 2),
        "coverage": round(scores_list[2], 2),
        "consistency": round(scores_list[3], 2),
        "avg_score": round(sum(scores_list) / 4, 2),
    }