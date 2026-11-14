# -*- coding: utf-8 -*-
"""Детерминированная генерация данных варианта 6 (seed = 10116).

Портфель проектов: капитальные затраты, эффекты, зависимости «i требует j»,
группа цифровых проектов G (требуют ИТ-платформы с фикс. затратами F),
капитальный бюджет. Результат: data/projects.csv + data/meta.json.
Все денежные величины — в млн руб.
"""
import json
import csv
import numpy as np

SEED = 10116
N_BASE = 15          # проектов в базовом наборе (вариант 6)
N_DEPS_BASE = 5      # зависимостей в базовом наборе (вариант 6)
PLATFORM_F = 40.0     # фиксированные затраты ИТ-платформы, млн руб.
G_SIZE = 3            # размер группы цифровых проектов


def generate(n=N_BASE, n_deps=None):
    """Генерация набора из n проектов. Один и тот же seed даёт одинаковые данные.

    Первые n значений cost и ratio совпадают для всех n >= 10 (выборки не зависят
    от n), поэтому экземпляры 10/20/40/80 вложены друг в друга — удобно для
    эксперимента масштабирования.
    """
    rng = np.random.default_rng(SEED)
    if n_deps is None:
        n_deps = max(3, n // 3)      # база: 15 проектов -> 5 зависимостей
    cost = rng.integers(18, 95, size=n).astype(float)      # кап. затраты, млн руб.
    ratio = rng.uniform(0.7, 2.0, size=n)                  # эффект/затраты
    # цифровые проекты (последние 3): заведомо привлекательные
    ratio[n - G_SIZE:] = rng.uniform(1.6, 2.4, size=G_SIZE)
    profit = np.round(cost * ratio, 1)                     # NPV, млн руб.
    # зависимости «i требует j», всегда j < i => ациклично; 1-нумерация
    deps = set()
    guard = 0
    while len(deps) < min(n_deps, n * (n - 1) // 2) and guard < 1000:
        i = int(rng.integers(1, n))
        j = int(rng.integers(0, i))
        deps.add((i + 1, j + 1))
        guard += 1
    deps = sorted(deps)
    g = list(range(n - G_SIZE + 1, n + 1))                    # 1-нумерация группы G
    budget = float(np.round(0.55 * cost.sum() / 5.0) * 5.0)  # ~55% сумм. затрат, кратно 5
    return dict(n=n, cost=cost, profit=profit, deps=deps, group=g,
                platform_cost=PLATFORM_F, budget=budget)


def save(d, csv_path='data/projects.csv', meta_path='data/meta.json'):
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['project', 'cost_mln', 'profit_mln', 'is_digital'])
        for i in range(d['n']):
            w.writerow([i + 1, d['cost'][i], d['profit'][i],
                        int(i + 1 in d['group'])])
    meta = dict(seed=SEED, n=d['n'], budget_mln=d['budget'],
                platform_cost_mln=d['platform_cost'],
                group_G=d['group'], deps=[list(p) for p in d['deps']],
                units='млн руб.')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    d = generate()
    save(d)
    print(f"n={d['n']}, бюджет={d['budget']} млн, платформа F={PLATFORM_F} млн, "
          f"G={d['group']}, зависимости={d['deps']}")
    print(f"сумма затрат={d['cost'].sum():.0f}, сумма эффектов={d['profit'].sum():.1f}")
