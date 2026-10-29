import json
import requests


def generate_test_cases(prompt_text, model="qwen2.5:1.5b", num_cases=5):
  url = "http://localhost:11434/api/generate"

  system_prompt = (
      "Ты профессиональный QA-инженер. Сгенерируй строго валидный "
      f"JSON-массив из {num_cases} объектов без лишнего текста и комментариев. "
      "Каждый объект должен содержать ключи: 'description' (описание самого действия, "
      "например, 'Ввод валидного логина и пароля', а не конкретные тестовые данные), "
      "'expected_result' (ожидаемый результат), 'boundary_values' (граничные значения)."
  )

  payload = {
      "model": model,
      "prompt": f"{system_prompt}\n\nЗадача: {prompt_text}",
      "format": "json",
      "stream": False,
      "options": {"num_predict": 1500, "temperature": 0.3},
  }

  response = requests.post(url, json=payload)
  if response.status_code == 200:
    try:
      raw_text = response.json()["response"]
      data = json.loads(raw_text)
      
      if isinstance(data, dict):
        for key in data:
          if isinstance(data[key], list):
            data = data[key]
            break

      normalized_cases = []
      for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
          continue

        str_values = [
            str(v) for v in item.values() if isinstance(v, str) and len(v.strip()) > 2
        ]

        desc = (
            item.get("description")
            or item.get("описание")
            or item.get("описание_теста")
            or item.get("title")
            or item.get("название")
            or (str_values[0] if len(str_values) > 0 else "Тестовый сценарий авторизации")
        )
        exp = (
            item.get("expected_result")
            or item.get("ожидаемый_результат")
            or item.get("результат")
            or item.get("expected")
            or (str_values[1] if len(str_values) > 1 else "Успешное выполнение действия")
        )
        bound = (
            item.get("boundary_values")
            or item.get("граничные_значения")
            or item.get("границы")
            or item.get("boundaries")
            or (str_values[2] if len(str_values) > 2 else "Стандартные граничные значения")
        )

        normalized_cases.append({
            "description": desc,
            "expected_result": exp,
            "boundary_values": bound,
        })
      
      return normalized_cases
    except json.JSONDecodeError:
      print("Ошибка парсинга JSON ответа модели.")
      return []
  else:
    raise Exception(f"Ошибка обращения к Ollama API: {response.text}")