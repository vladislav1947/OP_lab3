# -*- coding: utf-8 -*-
"""PNG-графики лабораторной (matplotlib, backend Agg — дисплея нет).

Все графики сохраняются в output/; plt.show() не вызывается.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(rows, path='output/scenarios.png'):
    """Зазор LP/MILP по трём сценариям."""
    x = np.arange(len(rows))
    lp = [r['lp_obj_mln'] for r in rows]
    milp = [r['obj_mln'] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.19, lp, 0.38, label='LP-релаксация (оценка сверху)', color='#8fb3d9')
    ax.bar(x + 0.19, milp, 0.38, label='MILP (оптимум)', color='#2d5f8a')
    for i, r in enumerate(rows):
        ax.annotate(f"gap {r['gap_pct']:.1f}%",
                    (x[i], max(lp[i], milp[i]) + 8), ha='center', fontsize=9)
        ax.annotate(f"проектов: {r['n_chosen']}\nзатраты {r['spent_mln']:.0f}",
                    (x[i] + 0.19, milp[i] * 0.55), ha='center', fontsize=7.5,
                    color='white')
    ax.set_xticks(x, [r['scenario'] for r in rows], fontsize=9)
    ax.set_ylabel('Суммарный эффект, млн руб.')
    ax.set_title('Сценарии бюджета: LP-оценка против MILP-оптимума')
    ax.set_ylim(0, max(lp) * 1.12)
    ax.legend()
    _save(fig, path)


def plot_scaling(rows, path='output/scaling.png'):
    """Время MILP, узлы ветвления и интегральный зазор от масштаба n."""
    ns = [r['n'] for r in rows]
    t = [r['milp_time_s'] for r in rows]
    nodes = [r['nodes'] for r in rows]
    gaps = [r['gap_pct'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].plot(ns, t, 'o-', color='#2d5f8a')
    axes[0].set_title('Время решения MILP')
    axes[0].set_ylabel('с (медиана из 5 прогонов)')
    axes[1].plot(ns, nodes, 'o-', color='#b3452d')
    axes[1].set_title('Узлы ветвления HiGHS')
    axes[2].plot(ns, gaps, 'o-', color='#3d7a3d')
    axes[2].set_title('Интегральный зазор LP, %')
    for ax in axes:
        ax.set_xlabel('число проектов n')
        ax.grid(alpha=0.3)
    if max(t) / max(min(t), 1e-9) > 30:
        axes[0].set_yscale('log')
    _save(fig, path)


def plot_bigm(rows, path='output/bigm.png'):
    """Эффект завышенной константы big-M: LP-оценка и зазор."""
    ms = [r['M'] for r in rows]
    lp = [r['lp_obj_mln'] for r in rows]
    gaps = [r['gap_pct'] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.semilogx(ms, lp, 'o-', color='#2d5f8a', label='LP-оценка сверху')
    ax1.set_xlabel('константа big-M (лог. шкала); точное значение M=3')
    ax1.set_ylabel('LP-оценка, млн руб.', color='#2d5f8a')
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.semilogx(ms, gaps, 's--', color='#b3452d', label='интегральный зазор, %')
    ax2.set_ylabel('зазор LP→MILP, %', color='#b3452d')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower right')
    ax1.set_title('Неоправданно большой big-M ослабляет LP-релаксацию')
    _save(fig, path)


def plot_formulations(rows, path='output/formulations.png'):
    """Сравнение формулировок agg/disagg: LP-оценка и узлы ветвления."""
    cases = sorted({r['case'] for r in rows})
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title, col in [
            (axes[0], 'lp_obj_mln', 'LP-оценка сверху (меньше = сильнее)', '#2d5f8a'),
            (axes[1], 'nodes', 'Узлы ветвления', '#b3452d')]:
        agg = [next(r[key] for r in rows if r['case'] == c and r['formulation'] == 'agg')
               for c in cases]
        dis = [next(r[key] for r in rows if r['case'] == c and r['formulation'] == 'disagg')
               for c in cases]
        ax.bar(x - 0.19, agg, 0.38, label='agg (big-M, слабая)', color=col, alpha=0.55)
        ax.bar(x + 0.19, dis, 0.38, label='disagg (сильная)', color=col)
        ax.set_xticks(x, cases, fontsize=9)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    _save(fig, path)
