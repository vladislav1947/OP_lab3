# -*- coding: utf-8 -*-
"""Обязательные эксперименты 1–6 и дополнительное задание.

Функции возвращают данные (словари/списки словарей); файловый вывод
(CSV, PNG) сосредоточен в main.py. Время — time.perf_counter.
"""
import time

import numpy as np

import data_gen
import model
import solve
import verify


def gap_pct(z_milp, z_lp):
    """Интегральный зазор: (LP-оценка сверху − оптимум MILP)/оптимум MILP, %."""
    return abs(z_lp - z_milp) / abs(z_milp) * 100.0


def chosen_set(m, x):
    return [i + 1 for i in range(m['n']) if round(x[i]) == 1]


# --- эксперимент 1: LP-релаксация против MILP -------------------------------

def exp1_lp_vs_milp(d, log_milp=None, log_lp=None):
    m = model.build(d)
    lp = solve.solve_lp(m, log_path=log_lp)
    mi = solve.solve_milp(m, log_path=log_milp)
    frac = [f'x{i + 1}={v:.3f}' for i, v in enumerate(lp['x'][:-1])
            if 1e-6 < abs(v - round(v))]
    return dict(m=m, lp=lp, milp=mi, gap=gap_pct(mi['obj'], lp['obj']),
                fractional=frac, y_lp=float(lp['x'][-1]),
                chosen=chosen_set(m, mi['x']))


# --- эксперимент 3: три сценария (базовый / дефицитный / стрессовый) --------

def exp3_scenarios(d, log_path=None):
    b0 = float(d['budget'])
    rows = []
    for name, b in [('базовый', b0), ('дефицитный (−20% бюджета)', 0.8 * b0),
                    ('стрессовый (−45% бюджета)', 0.55 * b0)]:
        m = model.build(d, budget=b)
        lp = solve.solve_lp(m)
        mi = solve.solve_milp(m, log_path=log_path)
        chosen = chosen_set(m, mi['x'])
        spent = float(np.dot(d['cost'], np.round(mi['x'][:m['n']]))
                     + d['platform_cost'] * round(mi['x'][-1]))
        rows.append(dict(scenario=name, budget_mln=b, chosen=', '.join(map(str, chosen)),
                         y=int(round(mi['x'][-1])), n_chosen=len(chosen),
                         spent_mln=spent, obj_mln=mi['obj'], lp_obj_mln=lp['obj'],
                         gap_pct=gap_pct(mi['obj'], lp['obj']),
                         nodes=mi['node_count'], time_s=mi['time']))
    return rows


# --- эксперимент 4: кривая «время решения / зазор» при масштабировании ------

def exp4_scaling(ns=(10, 20, 40, 80, 160), repeats=15):
    rows = []
    for n in ns:
        d_n = data_gen.generate(n)
        m = model.build(d_n)
        lp = solve.solve_lp(m)
        res = [solve.solve_milp(m) for _ in range(repeats)]
        times = sorted(r['time'] for r in res)
        mi = res[-1]
        rows.append(dict(n=n, deps=len(d_n['deps']), budget_mln=m['budget'],
                         lp_time_s=lp['time'], lp_obj_mln=lp['obj'],
                         milp_time_s=times[len(times) // 2],
                         milp_time_max_s=max(times), nodes=mi['node_count'],
                         obj_mln=mi['obj'], gap_pct=gap_pct(mi['obj'], lp['obj'])))
    return rows


# --- эксперимент 5: жадная эвристика ----------------------------------------

def greedy(d, budget=None):
    """Жадная эвристика: проекты по убыванию p_i/c_i; при выборе достраиваются
    все пререквизиты (транзитивно) и платформа, если взят цифровой проект.
    Отбрасывает вариант, если суммарные затраты превышают бюджет.
    """
    n = d['n']
    cost = np.asarray(d['cost'], dtype=float)
    profit = np.asarray(d['profit'], dtype=float)
    prereq = {i: j for i, j in d['deps']}   # i требует j, 1-нумерация
    g = set(d['group'])
    F = float(d['platform_cost'])
    b = float(d['budget'] if budget is None else budget)
    t0 = time.perf_counter()
    x = np.zeros(n, dtype=int)
    y = 0
    spent = 0.0
    for i0 in np.argsort(-(profit / cost)):
        i = int(i0) + 1
        if x[i0]:
            continue
        need, stack = set(), [i]           # замыкание пререквизитов; циклов нет (j<i)
        while stack:
            k = stack.pop()
            if k in need:
                continue
            need.add(k)
            if k in prereq:
                stack.append(prereq[k])
        add = [k for k in need if not x[k - 1]]
        if not add:
            continue
        add_cost = float(sum(cost[k - 1] for k in add))
        need_platform = any(k in g for k in add) and not y
        if need_platform:
            add_cost += F
        if spent + add_cost <= b:
            for k in add:
                x[k - 1] = 1
                spent += cost[k - 1]
            if need_platform:
                y = 1
                spent += F   # фикс. затраты платформы тоже расходуют бюджет
    return dict(x=x, y=y, obj=float(np.dot(profit, x)),
                time=time.perf_counter() - t0)


def exp5_greedy(d, milp):
    """Жадная против MILP. Время MILP — честное: без записи лога и с разогревом,
    медиана из 5 прогонов (первый вызов несёт накладные расходы HiGHS)."""
    gr = greedy(d)
    v = verify.verify(d, model.build(d), np.append(gr['x'], gr['y']))
    res = [solve.solve_milp(model.build(d)) for _ in range(5)]
    times = sorted(r['time'] for r in res)
    chosen = [i + 1 for i in range(d['n']) if gr['x'][i]]
    return dict(greedy_chosen=', '.join(map(str, chosen)), y=int(gr['y']),
                greedy_obj_mln=gr['obj'], greedy_time_s=gr['time'],
                milp_obj_mln=milp['obj'], milp_time_s=times[len(times) // 2],
                obj_gap_pct=(milp['obj'] - gr['obj']) / milp['obj'] * 100.0,
                greedy_feasible=v['ok'])


# --- эксперимент 6: выбор константы big-M ------------------------------------

def exp6_bigm(d):
    rows = []
    for big_m in (3.0, 30.0, 300.0, 1e6, 1e9):
        m = model.build(d, M=big_m)
        lp = solve.solve_lp(m)
        mi = solve.solve_milp(m)
        rows.append(dict(M=big_m, lp_obj_mln=lp['obj'], lp_y=float(lp['x'][-1]),
                         gap_pct=gap_pct(mi['obj'], lp['obj']), obj_mln=mi['obj'],
                         nodes=mi['node_count'], time_s=mi['time'],
                         M_over_budget=big_m / m['budget']))
    return rows


# --- доп. задание: две формулировки по силе LP-релаксации -------------------

def exp_extra_formulations(n_large=40, repeats=5):
    """Слабая (agg: sum_G x <= M y) против сильной (disagg: x_j <= y).

    Сравнение на дефицитном сценарии базового набора (n=15) и на большом
    экземпляре n=40: LP-оценка, зазор, узлы, время (медиана из repeats).
    """
    d15 = data_gen.generate()
    d40 = data_gen.generate(n_large)
    rows = []
    cases = [('n=15, дефицитный бюджет', d15, 0.8 * float(d15['budget'])),
             (f'n={n_large}, дефицитный бюджет', d40, 0.8 * float(d40['budget']))]
    for tag, dd, b in cases:
        for form in ('agg', 'disagg'):
            m = model.build(dd, budget=b, formulation=form)
            lp = solve.solve_lp(m)
            res = [solve.solve_milp(m) for _ in range(repeats)]
            times = sorted(r['time'] for r in res)
            mi = res[-1]
            rows.append(dict(case=tag, formulation=form,
                             rows_A=m['A'].shape[0], lp_obj_mln=lp['obj'],
                             gap_pct=gap_pct(mi['obj'], lp['obj']),
                             nodes=mi['node_count'], obj_mln=mi['obj'],
                             milp_time_s=times[len(times) // 2],
                             milp_time_max_s=max(times)))
    return rows
