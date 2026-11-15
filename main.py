# -*- coding: utf-8 -*-
"""Конвейер ЛР № 3 (вариант 6): данные → модель → решение → независимая
проверка → эксперименты 1–6 + доп. задание → графики и CSV.

Запуск одной командой:  python main.py   (или .venv/bin/python main.py)
Все артефакты пишутся в data/ и output/.
"""
import csv
import os

import data_gen
import experiments
import plots
import solve
import verify


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # --- данные варианта (seed 10116, воспроизводимо) ----------------------
    d = data_gen.generate()
    data_gen.save(d)
    print(f"[данные] n={d['n']}, бюджет B={d['budget']:.0f} млн руб., "
          f"платформа F={d['platform_cost']:.0f} млн руб., G={d['group']}, "
          f"зависимости={d['deps']}")

    # --- эксперимент 1: LP-релаксация против MILP --------------------------
    e1 = experiments.exp1_lp_vs_milp(
        d, log_milp='output/solver_base.log', log_lp='output/solver_lp_base.log')
    lp, mi = e1['lp'], e1['milp']
    print(f"[MILP] статус: {mi['status_str']}; цель = {mi['obj']:.4f} млн руб.; "
          f"узлов ветвления = {mi['node_count']}; время = {mi['time']:.3f} с")
    print(f"[LP  ] цель = {lp['obj']:.4f} млн руб.; дробные переменные: "
          f"{', '.join(e1['fractional']) or '—'}; y* = {e1['y_lp']:.3f}; "
          f"интегральный зазор = {e1['gap']:.2f}%")
    print(f"[LP  ] теневая цена бюджета = {lp.get('budget_shadow', float('nan')):.4f} "
          f"млн эффекта на 1 млн руб. бюджета")

    # --- эксперимент 2: независимая проверка (чистый NumPy) ------------------
    v = verify.verify(d, e1['m'], mi['x'])
    report = verify.format_report(v)
    print(report)
    with open('output/verification.txt', 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    # --- таблица решения ----------------------------------------------------
    with open('output/solution.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['project', 'cost_mln', 'profit_mln', 'is_digital', 'selected'])
        for i in range(d['n']):
            w.writerow([i + 1, d['cost'][i], d['profit'][i],
                        int(i + 1 in d['group']), int(round(mi['x'][i]))])
        w.writerow(['платформа_y', d['platform_cost'], 0.0, '-', int(round(mi['x'][-1]))])
        w.writerow(['ИТОГО', v['spent'], v['obj'], '-', ''])

    # --- эксперимент 3: сценарии --------------------------------------------
    scen = experiments.exp3_scenarios(d, log_path='output/solver_scenarios.log')
    write_csv('output/scenarios.csv', scen,
              ['scenario', 'budget_mln', 'chosen', 'y', 'n_chosen', 'spent_mln',
               'obj_mln', 'lp_obj_mln', 'gap_pct', 'nodes', 'time_s'])
    plots.plot_scenarios(scen)
    print("[сценарии] записано output/scenarios.csv + output/scenarios.png")
    for r in scen:
        print(f"    {r['scenario']}: выбрано [{r['chosen']}], y={r['y']}, "
              f"цель {r['obj_mln']:.1f} (LP {r['lp_obj_mln']:.1f}, "
              f"gap {r['gap_pct']:.2f}%)")

    # --- эксперимент 4: масштабирование -------------------------------------
    scale = experiments.exp4_scaling()
    write_csv('output/exp_scaling.csv', scale,
              ['n', 'deps', 'budget_mln', 'lp_time_s', 'lp_obj_mln',
               'milp_time_s', 'milp_time_max_s', 'nodes', 'obj_mln', 'gap_pct'])
    plots.plot_scaling(scale)
    print("[масштаб] записано output/exp_scaling.csv + output/scaling.png")
    for r in scale:
        print(f"    n={r['n']:3d}: MILP {r['milp_time_s']:.3f} с, узлов "
              f"{r['nodes']}, gap {r['gap_pct']:.2f}%")

    # --- эксперимент 5: жадная эвристика ------------------------------------
    gr = experiments.exp5_greedy(d, mi)
    write_csv('output/exp_greedy.csv', [gr],
              ['greedy_chosen', 'y', 'greedy_obj_mln', 'greedy_time_s',
               'milp_obj_mln', 'milp_time_s', 'obj_gap_pct', 'greedy_feasible'])
    print(f"[жадная] выбрано [{gr['greedy_chosen']}], y={gr['y']}, "
          f"цель {gr['greedy_obj_mln']:.1f} против MILP {gr['milp_obj_mln']:.1f} "
          f"(отставание {gr['obj_gap_pct']:.2f}%); время {gr['greedy_time_s']:.5f} с "
          f"против {gr['milp_time_s']:.3f} с; допустимость: {gr['greedy_feasible']}")

    # --- эксперимент 6: выбор big-M ------------------------------------------
    bigm = experiments.exp6_bigm(d)
    write_csv('output/exp_bigm.csv', bigm,
              ['M', 'lp_obj_mln', 'lp_y', 'gap_pct', 'obj_mln', 'nodes',
               'time_s', 'M_over_budget'])
    plots.plot_bigm(bigm)
    print("[big-M] записано output/exp_bigm.csv + output/bigm.png")
    for r in bigm:
        print(f"    M={r['M']:.0e}: LP {r['lp_obj_mln']:.2f} (y*={r['lp_y']:.4f}), "
              f"gap {r['gap_pct']:.2f}%, узлов {r['nodes']}, "
              f"время {r['time_s']:.3f} с")

    # --- доп. задание: сила формулировок ------------------------------------
    forms = experiments.exp_extra_formulations()
    write_csv('output/exp_formulations.csv', forms,
              ['case', 'formulation', 'rows_A', 'lp_obj_mln', 'gap_pct',
               'nodes', 'obj_mln', 'milp_time_s', 'milp_time_max_s'])
    plots.plot_formulations(forms)
    print("[формулировки] записано output/exp_formulations.csv + "
          "output/formulations.png")
    for r in forms:
        print(f"    {r['case']}, {r['formulation']:6s}: LP {r['lp_obj_mln']:.2f}, "
              f"gap {r['gap_pct']:.2f}%, узлов {r['nodes']}, "
              f"время {r['milp_time_s']:.3f} с")

    print("\nГОТОВО. Артефакты: output/ (solution.csv, scenarios.csv, "
          "exp_*.csv, verification.txt, *.png, solver_*.log)")


if __name__ == '__main__':
    main()
