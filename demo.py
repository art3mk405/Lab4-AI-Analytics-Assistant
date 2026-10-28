"""Розрахунки та, за прапорцем --live, справжні запити до Ollama."""
from pathlib import Path
import argparse
import json
import pandas as pd
from analytics import load_data, metrics, compare_periods, save_json
from assistant import (OllamaClient, auto_eda, suggest_hypotheses,
                       explain_metric_change, executive_summary, format_analysis)

ROOT = Path(__file__).resolve().parent
BEFORE = ('2024-04-29', '2024-05-03')
AFTER = ('2024-05-06', '2024-05-10')


def run(live=False, model='qwen2.5:7b', only=None):
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    users, payments, rejected, audit = load_data(ROOT/'data')
    rejected.to_csv(results/'rejected_payments.csv', index=False)
    save_json(audit, results/'audit.json')
    summary = metrics(payments)
    comparison = compare_periods(payments, BEFORE, AFTER)
    save_json(summary, results/'metrics.json')
    save_json({k:v for k,v in comparison.items() if k != 'breakdown'}, results/'comparison.json')
    for dimension, table in comparison['breakdown'].items():
        table.to_csv(results/f'change_by_{dimension}.csv', index=False)
    print('Платежі:', len(payments), '| Користувачі:', len(users), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    client = OllamaClient(model=model, log_dir=results/'llm_runs') if live else None
    reports = {}
    jobs = [
        ('eda', lambda: auto_eda(ROOT/'data/payments.csv', client)['analysis']),
        ('hypotheses', lambda: suggest_hypotheses(payments, comparison, client)),
        ('anomaly', lambda: explain_metric_change(payments, BEFORE, AFTER, client=client)['analysis']),
        ('summary', lambda: executive_summary(payments, comparison, client)),
    ]
    failures = []
    for name, job in jobs:
        if only and name != only:
            continue
        print(f'\n{name}: '+('Ollama обробляє запит…' if live else 'розрахунки без LLM'), flush=True)
        try:
            report = job()
            reports[name] = report
            save_json(report, results/f'{name}_{"live" if live else "computed"}.json')
            (results/f'{name}_{"live" if live else "computed"}.txt').write_text(format_analysis(report), encoding='utf-8')
            print(format_analysis(report), flush=True)
        except Exception as exc:
            failures.append({'task': name, 'error': str(exc)})
            print(f'ПОМИЛКА {name}: {exc}', flush=True)
    save_json({'live': live, 'model': model if live else None, 'only': only,
               'completed': list(reports), 'failures': failures}, results/'last_run.json')
    if failures:
        raise RuntimeError('Не всі запити завершилися. Дивись results/last_run.json і журнали llm_runs.')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Виконати справжні локальні LLM-запити')
    parser.add_argument('--model', default='qwen2.5:7b')
    parser.add_argument('--only', choices=['eda','hypotheses','anomaly','summary'])
    args = parser.parse_args()
    run(args.live, args.model, args.only)
