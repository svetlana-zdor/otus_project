import json
import os
import requests


def generate_quality_dataset(model="qwen2.5:1.5b", output_path="data/dataset.json"):
  os.makedirs("data", exist_ok=True)
  url = "http://localhost:11434/api/generate"

  prompt = """
Сгенерируй JSON-массив из 6 тест-кейсов для обучения модели оценки качества.
3 тест-кейса должны быть ХОРОШИМИ (подробное описание, четкий ожидаемый результат, указаны граничные значения, высокий балл).
3 тест-кейса должны быть ПЛОХИМИ (размытое описание, нет ожидаемого результата или граничных значений, низкий балл).

Верни строго JSON со следующей структурой каждого объекта:
[
  {
    "description": "строка с описанием",
    "expected_result": "строка с результатом",
    "boundary_values": "граничные значения",
    "quality_type": "good" или "bad",
    "readability": число от 0.0 до 1.0,
    "clarity": число от 0.0 до 1.0,
    "coverage": число от 0.0 до 1.0,
    "consistency": число от 0.0 до 1.0
  }
]
"""

  payload = {
      "model": model,
      "prompt": prompt,
      "format": "json",
      "stream": False,
      "options": {"num_predict": 2000, "temperature": 0.4},
  }

  print(f"Генерация обучающего датасета (хорошие/плохие кейсы) через {model}...")
  response = requests.post(url, json=payload)

  dataset = []
  if response.status_code == 200:
    try:
      raw_text = response.json()["response"]
      data = json.loads(raw_text)
      if isinstance(data, dict):
        for key in data:
          if isinstance(data[key], list):
            dataset = data[key]
            break
      elif isinstance(data, list):
        dataset = data
    except json.JSONDecodeError:
      print("Ошибка парсинга JSON при генерации датасета.")

  # Фолбэк-датасет, если LLM вернула некорректный JSON
  if not dataset:
    dataset = [
        {
            "description": (
                "Ввод валидного email и пароля от 8 символов на странице логина"
            ),
            "expected_result": (
                "Успешный вход, перенаправление в личный кабинет"
            ),
            "boundary_values": "Длина пароля: 8 символов, 64 символа",
            "quality_type": "good",
            "readability": 0.95,
            "clarity": 0.9,
            "coverage": 0.85,
            "consistency": 0.95,
        },
        {
            "description": "Тест кнопки",
            "expected_result": "Работает",
            "boundary_values": "Нет",
            "quality_type": "bad",
            "readability": 0.2,
            "clarity": 0.3,
            "coverage": 0.1,
            "consistency": 0.2,
        },
    ]

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

  print(f"Датасет сохранён в `{output_path}` (всего примеров: {len(dataset)})")
  return dataset