import json
import os
import re
import requests


def generate_quality_dataset(model="qwen2.5:1.5b", output_path="data/dataset.json"):
  os.makedirs("data", exist_ok=True)
  url = "http://localhost:11434/api/generate"

  prompt = """
Сгенерируй строго JSON-массив из 6 тест-кейсов для обучения модели оценки качества.
Не пиши никакого вводного текста, приветствий или пояснений. Верни ТОЛЬКО валидный JSON-массив.
3 тест-кейса должны быть ХОРОШИМИ (подробное описание, четкий ожидаемый результат, указаны граничные значения, высокий балл).
3 тест-кейса должны быть ПЛОХИМИ (размытое описание, нет ожидаемого результата или граничных значений, низкий балл).

Структура каждого объекта:
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
      "options": {"num_predict": 2500, "temperature": 0.3},
  }

  print(f"Генерация обучающего датасета (хорошие/плохие кейсы) через {model}...")
  response = requests.post(url, json=payload)

  dataset = []
  if response.status_code == 200:
    try:
      raw_text = response.json()["response"].strip()
      
      # Пытаемся найти валидный JSON-массив внутри ответа с помощью регулярного выражения
      match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
      if match:
        json_str = match.group(0)
      else:
        # Запасной вариант очистки через markdown
        json_str = raw_text
        if "```json" in json_str:
          json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
          json_str = json_str.split("```")[1].split("```")[0]

      dataset = json.loads(json_str.strip())
      
      if isinstance(dataset, dict):
        for key in dataset:
          if isinstance(dataset[key], list):
            dataset = dataset[key]
            break
      elif not isinstance(dataset, list):
        dataset = []
        
    except Exception as e:
      print(f"Ошибка парсинга JSON при генерации датасета: {e}")

  # Фолбэк-датасет, если LLM всё же вернула некорректный ответ
  if not dataset or len(dataset) < 4:
    print("Внимание: сработал фолбэк-датасет!")
    dataset = [
        {
            "description": "Ввод валидного email и пароля от 8 символов на странице логина",
            "expected_result": "Успешный вход, перенаправление в личный кабинет",
            "boundary_values": "Длина пароля: 8 символов, 64 символа",
            "quality_type": "good",
            "readability": 0.95,
            "clarity": 0.9,
            "coverage": 0.85,
            "consistency": 0.95,
        },
        {
            "description": "Некорректный ввод данных авторизации без заполнения полей",
            "expected_result": "Отображение ошибок валидации под полями формы",
            "boundary_values": "Пустые строки",
            "quality_type": "good",
            "readability": 0.9,
            "clarity": 0.85,
            "coverage": 0.8,
            "consistency": 0.9,
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
        {
            "description": "Проверить поле",
            "expected_result": "Ок",
            "boundary_values": "нет",
            "quality_type": "bad",
            "readability": 0.1,
            "clarity": 0.2,
            "coverage": 0.1,
            "consistency": 0.1,
        }
    ]

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

  print(f"Датасет успешно сохранён в `{output_path}` (всего примеров: {len(dataset)})")
  return dataset