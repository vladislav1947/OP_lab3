# -*- coding: utf-8 -*-
"""Построение MILP-модели варианта 6 в матричной форме A z <= b.

Переменные (все бинарные): z = (x_1, ..., x_n, y), где y — развёртывание
ИТ-платформы. Модель: max p^T x при ограничениях бюджета, зависимостей
«i требует j» и платформенной связки.

Формулировки платформенной связки:
  agg:    sum_{j in G} x_j <= M * y   — big-M связка, минимально допустимое
                                      (точное) значение M = |G|;
  disagg: x_j <= y для каждого j в G  — дезагрегированная (сильная), для
                                      сравнения в доп. задании.
Для целочисленных решений формулировки эквивалентны.
"""
import numpy as np


def build(d, budget=None, formulation='agg', M=None):
    """Матрицы задачи по данным d (словарь из data_gen.generate)."""
    n = int(d['n'])
    cost = np.asarray(d['cost'], dtype=float)
    profit = np.asarray(d['profit'], dtype=float)
    deps = d['deps']
    g = d['group']
    F = float(d['platform_cost'])
    B = float(d['budget'] if budget is None else budget)
    if M is None:
        M = len(g)  # точная (минимально допустимая) константа big-M

    rows, rhs = [], []

    def row(coef_x, coef_y, b):
        rows.append(np.append(np.asarray(coef_x, dtype=float), coef_y))
        rhs.append(float(b))

    row(cost, F, B)                      # капитальный бюджет: c^T x + F y <= B
    for i, j in deps:                    # зависимости: x_i - x_j <= 0
        r = np.zeros(n)
        r[i - 1], r[j - 1] = 1.0, -1.0
        row(r, 0.0, 0.0)
    if formulation == 'agg':             # big-M связка платформы
        r = np.zeros(n)
        for j in g:
            r[j - 1] = 1.0
        row(r, -float(M), 0.0)
    elif formulation == 'disagg':        # сильная (дезагрегированная) связка
        for j in g:
            r = np.zeros(n)
            r[j - 1] = 1.0
            row(r, -1.0, 0.0)
    else:
        raise ValueError(f'неизвестная формулировка: {formulation}')

    # max p^T x; платформа y собственного эффекта не приносит
    c = np.append(profit, 0.0)
    return dict(A=np.array(rows), b=np.array(rhs), c=c, n=n, budget=B,
                formulation=formulation, M=(float(M) if formulation == 'agg' else None))
