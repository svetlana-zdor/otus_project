import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_report(test_cases, evaluations):
  os.makedirs("outputs", exist_ok=True)

  report_data = []
  success_count = 0

  for i, case in enumerate(test_cases):
    eval_res = evaluations[i]
    is_success = eval_res["avg_score"] >= 0.5
    if is_success:
      success_count += 1

    description = (
        case.get("description")
        or case.get("описание")
        or case.get("описание_теста")
        or "Тестовый сценарий"
    )
    expected_result = (
        case.get("expected_result")
        or case.get("ожидаемый_результат")
        or ""
    )
    boundary_values = (
        case.get("boundary_values")
        or case.get("граничные_значения")
        or ""
    )

    report_data.append({
        "id": i + 1,
        "description": description,
        "expected_result": expected_result,
        "boundary_values": boundary_values,
        "quality_status": "Успешно" if is_success else "Требует доработки",
        "score": eval_res["avg_score"],
    })

  # 1. Сохранение сырых результатов в JSON
  with open(
      "outputs/test_results.json", "w", encoding="utf-8"
  ) as f:
    json.dump(report_data, f, ensure_ascii=False, indent=4)

  # 2. Визуализация метрик (matplotlib + seaborn)
  df = pd.DataFrame(evaluations)
  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(7, 4))
  sns.barplot(data=df, palette="Blues_d")
  plt.title("Распределение качества тест-кейсов по метрикам")
  plt.ylim(0, 1.0)
  plt.ylabel("Оценка (0-1)")
  plt.savefig("outputs/metrics_distribution.png", bbox_inches="tight")
  plt.close()

  # 3. Генерация отчёта в формате Markdown
  total_count = len(test_cases)
  failed_count = total_count - success_count

  md_content = f"""# Отчёт о тестировании сценариев

## Общая информация
- Количество сценариев: {total_count}
- Успешные: {success_count}
- Неуспешные: {failed_count}

## Распределение качества
![График распределения метрик](metrics_distribution.png)

## Список сценариев
| Описание | Ожидаемый результат | Качество | Оценка |
| --- | --- | --- | --- |
"""
  for item in report_data:
    md_content += (
        f"| {item['description']} | {item['expected_result']} | {item['quality_status']} |"
        f" {item['score']} |\n"
    )

  with open("outputs/report.md", "w", encoding="utf-8") as f:
    f.write(md_content)