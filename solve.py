# -*- coding: utf-8 -*-
"""Решение MILP и LP-релаксации солвером HiGHS.

MILP — scipy.optimize.milp, LP — scipy.optimize.linprog(method='highs').
Лог HiGHS печатается из C++ прямо в stdout, поэтому для его захвата
перенаправляется файловый дескриптор 1 (contextlib.redirect_stdout его
не ловит). Время — perf_counter вокруг вызова солвера.
"""
import contextlib
import os
import sys
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

STATUS = {0: 'optimal', 1: 'limit (итерации/время/узлы)', 2: 'infeasible',
          3: 'unbounded', 4: 'другое'}


@contextlib.contextmanager
def _fd_stdout(path):
    """Перенаправление fd 1 в файл (для перехвата C++-вывода HiGHS)."""
    sys.stdout.flush()
    saved = os.dup(1)
    f = open(path, 'a')
    os.dup2(f.fileno(), 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        f.close()


def solve_milp(m, log_path=None, time_limit=60.0):
    """Решает MILP: min -c^T z при A z <= b, z — бинарные. Возвращает словарь."""
    n1 = len(m['c'])
    con = LinearConstraint(m['A'], -np.inf, m['b'])
    opts = dict(disp=bool(log_path), time_limit=time_limit, presolve=True)
    t0 = time.perf_counter()
    with _fd_stdout(log_path) if log_path else contextlib.nullcontext():
        res = milp(c=-m['c'], constraints=con, integrality=np.ones(n1),
                   bounds=Bounds(0, 1), options=opts)
    dt = time.perf_counter() - t0
    x = np.asarray(res.x) if res.x is not None else np.full(n1, np.nan)
    # пересчёт цели напрямую по вектору решения (независимо от res.fun)
    obj = float(np.dot(m['c'], x)) if res.x is not None else float('nan')
    dual = -float(res.mip_dual_bound)  # верхняя оценка максимума
    return dict(status=res.status, status_str=STATUS.get(res.status, '?'),
                message=res.message, x=x, obj=obj, fun=float(res.fun),
                node_count=int(res.mip_node_count), dual_bound=dual,
                solver_gap=float(res.mip_gap), time=dt)


def solve_lp(m, log_path=None):
    """LP-релаксация: те же A, b, но 0 <= z <= 1 без целочисленности."""
    t0 = time.perf_counter()
    opts = dict(disp=bool(log_path))  # лог HiGHS пишем только если он нужен
    with _fd_stdout(log_path) if log_path else contextlib.nullcontext():
        res = linprog(-m['c'], A_ub=m['A'], b_ub=m['b'], bounds=(0, 1),
                      method='highs', options=opts)
    dt = time.perf_counter() - t0
    x = np.asarray(res.x) if res.x is not None else np.full(len(m['c']), np.nan)
    obj = float(np.dot(m['c'], x)) if res.x is not None else float('nan')
    out = dict(status=res.status, status_str=STATUS.get(res.status, '?'),
               x=x, obj=obj, fun=float(res.fun), time=dt)
    # теневая цена бюджета (1-я строка A): marginals в scipy имеют знак задачи min
    if res.status == 0 and m['A'].shape[0] > 0:
        out['budget_shadow'] = float(-res.ineqlin.marginals[0])
    return out
