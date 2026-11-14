# -*- coding: utf-8 -*-
"""Независимая проверка решения MILP на чистом NumPy.

Проверка работает ТОЛЬКО по массивам данных варианта (cost, profit, deps,
group, F, бюджет) и по вектору решения z = (x_1..x_n, y): объект солвера и
его результаты не используются. Пересчитываются все ограничения (бюджет,
зависимости, платформенная big-M связка), бинарность и целевая функция.
"""
import numpy as np

TOL = 1e-7      # допуск допустимости
INT_TOL = 1e-6  # допуск целочисленности


def verify(d, m, z):
    """Возвращает словарь: поимённые проверки, итог ok, пересчитанные величины."""
    n = int(d['n'])
    cost = np.asarray(d['cost'], dtype=float)
    profit = np.asarray(d['profit'], dtype=float)
    F = float(d['platform_cost'])
    B = float(m['budget'])
    g = d['group']
    deps = d['deps']
    z = np.asarray(z, dtype=float)

    frac_dev = float(np.max(np.abs(z - np.round(z))))
    x = np.round(z[:n])
    y = float(np.round(z[n]))

    spent = float(np.dot(cost, x) + F * y)          # капитальные затраты портфеля
    obj = float(np.dot(profit, x))                  # пересчёт целевой функции

    checks = {}
    checks['целочисленность (|z-round(z)| < 1e-6)'] = frac_dev < INT_TOL
    checks['бинарность (все z в {0,1})'] = bool(
        np.all(np.isin(np.round(z), [0.0, 1.0])))
    checks[f'бюджет: {spent:.1f} <= {B:.1f} млн руб.'] = spent <= B + TOL
    checks[f'зависимости ({len(deps)}): x_i <= x_j'] = all(
        x[i - 1] <= x[j - 1] + TOL for i, j in deps)
    if m['formulation'] == 'agg':
        lhs = float(sum(x[j - 1] for j in g))
        checks[f'big-M связка: sum_G x = {lhs:.0f} <= M*y = {m["M"] * y:.0f}'] = \
            lhs <= m['M'] * y + TOL
    else:
        checks['платформенные связки: x_j <= y для j из G'] = all(
            x[j - 1] <= y + TOL for j in g)
    # матричная проверка «всех связок сразу»: A z <= b + tol
    viol = float(np.max(m['A'] @ z - m['b']))
    checks['все строки A z <= b (допуск 1e-7)'] = viol <= TOL

    return dict(ok=all(checks.values()), checks=checks, spent=spent, obj=obj,
                y=y, x=x, max_violation=viol, frac_dev=frac_dev)


def format_report(v):
    """Читаемая сводка проверки (для консоли и output/verification.txt)."""
    lines = ['НЕЗАВИСИМАЯ ПРОВЕРКА (чистый NumPy, без объекта солвера)']
    for name, ok in v['checks'].items():
        lines.append(f"  [{'OK' if ok else 'FAIL'}] {name}")
    lines.append(f"  пересчитанная цель: {v['obj']:.4f} млн руб.")
    lines.append(f"  максимальное нарушение ограничений: {v['max_violation']:.2e}")
    lines.append(f"  отклонение от целочисленности: {v['frac_dev']:.2e}")
    lines.append(f"  ИТОГ: {'ПРОЙДЕНА' if v['ok'] else 'ПРОВАЛЕНА'}")
    return '\n'.join(lines)
