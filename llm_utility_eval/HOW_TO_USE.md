# 🚀 Как использовать LLM Utility Evaluation

## 📍 Где находится всё?

```
llm_utility_eval/
├── models.yaml              # 19 моделей с их характеристиками
├── config.yaml              # Веса для формулы (α, β, γ, δ)
├── main.py                  # Основной скрипт для быстрой оценки
├── generate_statistics.py   # Генерирует таблицы в консоли
├── generate_markdown_report.py  # Создает полный отчет
└── LLM_EVALUATION_REPORT.md    # 📊 ПОЛНЫЙ ОТЧЕТ С TOP-10
```

## 🏃 Как запустить?

### 1. Быстрая оценка
```bash
cd llm_utility_eval
source venv/bin/activate  # или создать: python3 -m venv venv
pip install pyyaml

# Показать топ-5 моделей
python main.py --top 5

# Использовать профиль "бюджетный"
python main.py --profile budget_conscious
```

### 2. Полная статистика в консоли
```bash
pip install tabulate
python generate_statistics.py
```

### 3. Создать полный отчет (РЕКОМЕНДУЕТСЯ!)
```bash
python generate_markdown_report.py
```

## 📊 Где смотреть результаты?

### Главный файл: `LLM_EVALUATION_REPORT.md`

Содержит:
- ✅ Все 19 моделей с ценами
- ✅ Формулу: Utility = α×Quality + β×Speed + γ×Stability − δ×Cost
- ✅ TOP-10 в каждой категории:
  - Default (Balanced) 
  - Quality First
  - Speed Optimized
  - Budget Conscious
- ✅ Рекомендации по использованию

## 💰 Стоимость моделей

| Категория цен | Модели | Примерная цена |
|---------------|--------|----------------|
| Очень дешево | DeepSeek-V3, GPT-4o-mini | $0.10-0.50 / 1M токенов |
| Дешево | Gemini Flash, Claude Haiku | $0.50-2.00 / 1M токенов |
| Средне | Claude 3.5 Sonnet, Llama 70B | $3.00-5.00 / 1M токенов |
| Дорого | Claude 4 Sonnet, Gemini Pro | $5.00-10.00 / 1M токенов |
| Очень дорого | GPT-4 Turbo, GPT-4o | $10.00-20.00 / 1M токенов |
| Премиум | Claude Opus, o1-preview | $30.00+ / 1M токенов |

## 🎯 Категории оценки

1. **Default (Balanced)** - Баланс всех факторов
   - Лидер: DeepSeek-V3 (2.56)

2. **Quality First** - Приоритет качества (α=1.5)
   - Лидер: Claude 3.5 Sonnet (2.705)

3. **Speed Optimized** - Приоритет скорости (β=1.5)
   - Лидер: DeepSeek-V3 (2.813)

4. **Budget Conscious** - Приоритет низкой цены (δ=2.0)
   - Лидер: DeepSeek-V3 (2.473)

## 🔧 Как изменить параметры?

### Добавить новую модель
Отредактируй `models.yaml`:
```yaml
- name: "New Model"
  quality: 0.85      # 0-1 (качество)
  speed: 0.90        # 0-1 (скорость)
  stability: 0.95    # 0-1 (стабильность)
  cost_penalty: 0.10 # 0-1 (чем больше, тем дороже)
```

### Изменить веса формулы
Отредактируй `config.yaml`:
```yaml
weights:
  alpha: 1.2    # Важность качества
  beta: 0.8     # Важность скорости
  gamma: 1.0    # Важность стабильности
  delta: 1.5    # Штраф за цену
```

## 📈 Экспорт результатов

```bash
# Сохранить в JSON
python main.py --export results.json

# Создать HTML версию (нужен markdown)
pip install markdown
# Затем generate_markdown_report.py создаст и HTML файл
```

---
Все TOP-10 рейтинги находятся в файле **LLM_EVALUATION_REPORT.md**!