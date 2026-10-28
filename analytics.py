"""Розрахунки асистента. Усі числові результати отримуються без LLM."""
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np


def read_csv(path):
    # NA — код Намібії, тому стандартний список NA тут не використовуємо.
    return pd.read_csv(path, keep_default_na=False, na_values=[''], low_memory=False)


def _required(df, columns):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f'Немає обов’язкових колонок: {sorted(missing)}')
    if df.empty:
        raise ValueError('Таблиця порожня.')


def load_data(folder):
    folder = Path(folder)
    users = read_csv(folder / 'registrations.csv')
    payments = read_csv(folder / 'payments.csv')
    countries = read_csv(folder / 'country_info_dict.csv')
    _required(users, ['user_id', 'reg_dt', 'app', 'country_code'])
    _required(payments, ['user_id', 'pay_dt', 'amount_usd', 'feature_type_id'])
    _required(countries, ['country_code', 'country_name'])
    users = users.copy()
    if users.user_id.isna().any() or users.user_id.duplicated().any():
        raise ValueError('user_id у registrations має бути заповненим і унікальним.')
    if countries.country_code.dropna().duplicated().any():
        raise ValueError('Довідник країн містить повторні ключі.')
    users['reg_dt'] = pd.to_datetime(users.reg_dt, errors='coerce')
    if users.reg_dt.isna().any():
        raise ValueError('У registrations є некоректна дата реєстрації.')
    users['country_code'] = users.country_code.fillna('Unknown')
    users['app'] = users.app.fillna('Unknown')
    raw_rows = len(payments)
    payments = payments.copy()
    payments['pay_dt'] = pd.to_datetime(payments.pay_dt, errors='coerce')
    payments['amount_usd'] = pd.to_numeric(payments.amount_usd, errors='coerce')
    reason = pd.Series('', index=payments.index, dtype='string')
    rules = [
        ('invalid_date', payments.pay_dt.isna()),
        ('invalid_amount', payments.amount_usd.isna() | ~np.isfinite(payments.amount_usd)),
        ('negative_amount', payments.amount_usd.lt(0)),
        ('unmatched_user', ~payments.user_id.isin(users.user_id)),
    ]
    for label, mask in rules:
        reason.loc[reason.eq('') & mask] = label
    rejected = payments.loc[reason.ne('')].copy()
    rejected['reason'] = reason.loc[reason.ne('')]
    payments = payments.loc[reason.eq('')].copy()
    if payments.empty:
        raise ValueError('Після перевірки не залишилося придатних платежів.')
    # Без transaction_id однакові платежі можуть бути окремими покупками.
    # Тому лише фіксуємо повні збіги, не видаляємо автоматично.
    duplicates = int(payments.duplicated().sum())
    rounded = (payments.amount_usd * 100).round()
    if not np.allclose(payments.amount_usd * 100, rounded, atol=1e-6, rtol=0):
        raise ValueError('Суми мають більше двох десяткових знаків; уточніть точність валюти.')
    payments['amount_cents'] = rounded.astype('int64')
    total_before = int(payments.amount_cents.sum())
    payments = payments.merge(users[['user_id', 'reg_dt', 'app', 'country_code']],
                              on='user_id', how='left', validate='many_to_one')
    country_map = countries.dropna(subset=['country_code']).set_index('country_code').country_name
    payments['country_name'] = payments.country_code.map(country_map).fillna('Unknown')
    assert total_before == int(payments.amount_cents.sum())
    audit = {'raw_payments': raw_rows, 'valid_payments': len(payments),
             'rejected_payments': len(rejected), 'exact_payment_duplicates': duplicates,
             'payments_before_registration': int((payments.pay_dt < payments.reg_dt).sum()),
             'missing_user_age': int(users.age.isna().sum()) if 'age' in users else None}
    return users, payments, rejected, audit


def profile_dataframe(df, id_columns=()):
    if df.empty:
        raise ValueError('Неможливо виконати EDA порожньої таблиці.')
    if not df.columns.is_unique:
        raise ValueError('Назви колонок повинні бути унікальними.')
    columns, distributions = [], {}
    for name in df.columns:
        s = df[name]
        columns.append({'column': str(name), 'dtype': str(s.dtype),
                        'missing': int(s.isna().sum()),
                        'missing_pct': round(float(s.isna().mean() * 100), 4),
                        'unique': int(s.nunique(dropna=True))})
        if name in id_columns:
            continue
        if pd.api.types.is_numeric_dtype(s):
            values = s.dropna()
            finite = values[np.isfinite(values)]
            if len(finite):
                q1, q3 = finite.quantile([.25, .75])
                iqr = q3 - q1
                distributions[str(name)] = {
                    'kind': 'numeric', 'min': float(finite.min()), 'max': float(finite.max()),
                    'mean': float(finite.mean()), 'median': float(finite.median()),
                    'q1': float(q1), 'q3': float(q3),
                    'nonfinite': int(len(values) - len(finite)),
                    'iqr_flags': int(((finite < q1-1.5*iqr) | (finite > q3+1.5*iqr)).sum()),
                    'histogram': {'counts': np.histogram(finite, bins=10)[0].tolist(),
                                  'edges': np.histogram(finite, bins=10)[1].tolist()}}
            else:
                distributions[str(name)] = {'kind': 'numeric', 'finite_count': 0}
        elif pd.api.types.is_datetime64_any_dtype(s):
            distributions[str(name)] = {'kind': 'date', 'min': str(s.min()), 'max': str(s.max())}
        elif s.nunique(dropna=True) <= 1000:
            distributions[str(name)] = {'kind': 'category', 'top': [
                {'label': str(k), 'count': int(v)} for k, v in s.value_counts().head(10).items()]}
    return {'rows': len(df), 'columns': columns, 'duplicates': int(df.duplicated().sum()),
            'distributions': distributions}


def period_slice(payments, start, end):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a.tzinfo is not None or b.tzinfo is not None:
        raise ValueError('Використайте дати без часового поясу, як у вихідних даних.')
    if a != a.normalize() or b != b.normalize() or a > b:
        raise ValueError('Потрібні календарні дати, початок не пізніше завершення.')
    if a < payments.pay_dt.min().normalize() or b > payments.pay_dt.max().normalize():
        raise ValueError('Період виходить за межі наявних даних.')
    selected = payments.loc[payments.pay_dt.ge(a) & payments.pay_dt.lt(b + pd.Timedelta(days=1))]
    return selected, a, b


def metrics(payments, start='2024-04-05', end='2024-05-10'):
    df, a, b = period_slice(payments, start, end)
    cents = int(df.amount_cents.sum())
    return {'start': str(a.date()), 'end': str(b.date()), 'days': (b-a).days+1,
            'payments': len(df), 'paying_users': int(df.user_id.nunique()),
            'revenue_usd': cents / 100,
            'avg_payment_usd': cents / 100 / len(df) if len(df) else None}


def compare_periods(payments, before=('2024-04-29', '2024-05-03'),
                    after=('2024-05-06', '2024-05-10'), metric='revenue_usd'):
    if metric not in ('revenue_usd', 'payments'):
        raise ValueError('Підтримано revenue_usd і payments. Для retention потрібна інша логіка.')
    old, a, b = period_slice(payments, *before)
    new, c, d = period_slice(payments, *after)
    if (b-a).days != (d-c).days:
        raise ValueError('Порівнюйте періоди однакової тривалості.')
    if not (b < c or d < a):
        raise ValueError('Періоди не повинні перетинатися.')
    if c <= a:
        raise ValueError('Новий період має бути пізнішим за базовий.')
    scale = 100 if metric == 'revenue_usd' else 1
    def aggregate(frame, dimension=None):
        if dimension:
            grouped = frame.groupby(dimension, dropna=False)
            return grouped.amount_cents.sum() if metric == 'revenue_usd' else grouped.size()
        return int(frame.amount_cents.sum()) if metric == 'revenue_usd' else len(frame)
    old_value, new_value = aggregate(old), aggregate(new)
    delta = new_value-old_value
    breakdown = {}
    for dimension in ['app', 'country_code', 'feature_type_id']:
        frame = pd.concat([aggregate(old, dimension).rename('before'),
                           aggregate(new, dimension).rename('after')], axis=1).fillna(0).astype('int64')
        frame['delta'] = frame.after-frame.before
        assert int(frame.delta.sum()) == delta
        frame['contribution_pct'] = frame.delta / delta * 100 if delta else np.nan
        for col in ['before', 'after', 'delta']:
            frame[col] = frame[col] / scale
        frame.index = frame.index.map(str)
        frame = frame.loc[frame.delta.abs().sort_values(ascending=False).index]
        breakdown[dimension] = frame.reset_index().rename(columns={dimension: 'segment'})
    return {'metric': metric, 'before_period': list(before), 'after_period': list(after),
            'before': old_value / scale, 'after': new_value / scale, 'delta': delta / scale,
            'change_pct': delta / old_value * 100 if old_value else None,
            'same_weekdays': a.weekday() == c.weekday(), 'breakdown': breakdown}


class Facts:
    def __init__(self):
        self.items = {}

    def add(self, label, value, unit=''):
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        elif isinstance(value, float):
            value = round(value, 2)
        key = f'F{len(self.items)+1:03d}'
        self.items[key] = {'label': label, 'value': value, 'unit': unit}
        return key


def profile_facts(profile):
    facts = Facts()
    facts.add('Кількість рядків', profile['rows'], 'рядків')
    facts.add('Повні повтори рядків; це не автоматичний доказ помилки', profile['duplicates'], 'рядків')
    for col in profile['columns']:
        facts.add(f"Пропуски у {col['column']}", col['missing'], 'рядків')
    for name, dist in profile['distributions'].items():
        if dist['kind'] == 'numeric' and 'mean' in dist:
            for key in ['mean', 'median', 'max', 'iqr_flags']:
                facts.add(f'{name}: {key}', dist[key])
            facts.add(f'{name}: частка позначок IQR від усіх рядків', dist['iqr_flags']/profile['rows']*100, '%')
    return facts.items


def business_facts(payments, comparison, start='2024-04-05', end='2024-05-10'):
    facts = Facts()
    old = metrics(payments, *comparison['before_period'])
    new = metrics(payments, *comparison['after_period'])
    period_note = f"база {comparison['before_period']}; новий період {comparison['after_period']}"
    facts.add('Періоди порівняння', period_note)
    for key in ['payments', 'paying_users', 'avg_payment_usd']:
        facts.add(f'{key}; база', old[key])
        facts.add(f'{key}; новий період', new[key])
    for key in ['before', 'after', 'delta', 'change_pct']:
        direction = ('; зменшення' if comparison[key] < 0 else '; збільшення' if comparison[key] > 0 else '; без зміни') if key == 'delta' else ''
        facts.add(f"{comparison['metric']}: {key}{direction}", comparison[key])
    for dimension, table in comparison['breakdown'].items():
        for row in table.head(5).to_dict('records'):
            for key in ['before', 'after', 'delta']:
                direction = ('; зменшення' if row[key] < 0 else '; збільшення' if row[key] > 0 else '; без зміни') if key == 'delta' else ''
                facts.add(f"{dimension}={row['segment']}; {comparison['metric']}: {key}{direction}", row[key])
    return facts.items


def save_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
