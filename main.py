import json
from src.evaluator import TestCaseQualityEvaluator
from src.generator import generate_test_cases
from src.reporter import generate_report

if __name__ == "__main__":
  print("=== Шаг 0. Загрузка эталонного датасета (хорошие/плохие кейсы) ===")
  try:
    with open("data/dataset.json", "r", encoding="utf-8") as f:
      dataset = json.load(f)
    print(f"Датасет успешно загружен. Всего примеров: {len(dataset)}")
  except FileNotFoundError:
    print("Ошибка: файл data/dataset.json не найден.")
    dataset = []

  if not dataset:
    print("Датасет пуст. Проверьте наличие файла.")
    exit(1)

  print("\n=== Шаг 0.1. Обучение NLP-модели (PyTorch) на датасете ===")
  evaluator = TestCaseQualityEvaluator()
  # Обучаем модель на датасете (увеличенное число эпох помогает лучше разделять оценки)[cite: 1]
  evaluator.train_on_dataset(dataset, epochs=10, lr=1e-4)

  print(
      "\n=== Шаг 1. Генерация целевых тест-кейсов через Ollama (qwen2.5:1.5b) ==="
  )
  cases = generate_test_cases(
      "Форма авторизации пользователя", model="qwen2.5:1.5b", num_cases=5
  )

  if not cases:
    print(
        "Не удалось сгенерировать тест-кейсы. Убедитесь, что Ollama запущена."
    )
    exit(1)

  print(
      "\n=== Шаг 2. Оценка качества с помощью обученной PyTorch модели ==="
  )
  evaluations = []

  for case in cases:
    text = f"{case.get('description', '')} {case.get('expected_result', '')}"
    evaluations.append(evaluator.evaluate_text(text))

  print("\n=== Шаг 3. Генерация JSON, графиков и Markdown отчёта ===")
  generate_report(cases, evaluations)

  print("\nВсе этапы успешно выполнены! Артефакты сохранены в папке `outputs/`.")