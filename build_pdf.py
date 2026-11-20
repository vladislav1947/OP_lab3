# -*- coding: utf-8 -*-
"""Сборка REPORT.md в PDF: markdown -> HTML -> weasyprint.

Запуск: .venv/bin/python build_pdf.py
Результат: ЛР3_Осипов_ВА_Б24-517.pdf (в корне репозитория).
"""
import markdown
from weasyprint import HTML, CSS

TITLE = ('Лабораторная работа № 3. MILP-модель дискретного выбора. '
         'Вариант 6. Осипов В. А., группа Б24-517')
OUT = 'ЛР3_Осипов_ВА_Б24-517.pdf'

STYLESHEET = CSS(string='''
@page { margin: 2cm; }
body { font-family: "DejaVu Sans"; max-width: 800px; margin: 2em auto;
       font-size: 11pt; line-height: 1.45; }
h1, h2, h3 { color: #1a3c6e; }
table { border-collapse: collapse; }
td, th { border: 1px solid #999; padding: 4px 8px; }
img { max-width: 100%; }
pre { background: #f5f5f5; padding: 8px; overflow-x: auto; }
''')


def build():
    with open('REPORT.md', encoding='utf-8') as f:
        md_text = f.read()
    # Заголовок PDF + содержимое отчёта
    html_body = markdown.markdown(f'# {TITLE}\n\n' + md_text,
                                  extensions=['tables', 'fenced_code'])
    html = ('<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{TITLE}</title></head><body>{html_body}</body></html>')
    # base_url='.' — относительные пути к PNG подтянутся из репозитория
    HTML(string=html, base_url='.').write_pdf(OUT, stylesheets=[STYLESHEET])
    print(f'PDF собран: {OUT}')


if __name__ == '__main__':
    build()
