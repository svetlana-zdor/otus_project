from src.dataset_generator import generate_quality_dataset
from src.evaluator import TestCaseQualityEvaluator
from src.generator import generate_test_cases
from src.reporter import generate_report

if __name__ == "__main__":
  print("=== Шаг 0. Генерация датасета с хорошими и плохими сценариями ===")
  dataset = generate_quality_dataset(model="qwen2.5:1.5b")

  print(
      "\n=== Шаг 1. Генерация тест-кейсов через Ollama (qwen2.5:1.5b) ==="
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
      "\n=== Шаг 2. Оценка качества с помощью PyTorch модели (rubert-tiny2)"
      " ==="
  )
  evaluator = TestCaseQualityEvaluator()
  evaluations = []

  for case in cases:
    text = f"{case.get('description', '')} {case.get('expected_result', '')}"
    evaluations.append(evaluator.evaluate_text(text))

  print("\n=== Шаг 3. Генерация JSON, графиков и Markdown отчёта ===")
  generate_report(cases, evaluations)

  print("\nВсе этапы успешно выполнены! Созданы папки `data/` и `outputs/`.")