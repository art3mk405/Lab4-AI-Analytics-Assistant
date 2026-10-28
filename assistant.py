"""Чотири функції асистента з локальним Ollama API та перевіркою відповіді."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from datetime import datetime, timezone
import json
import os
import re
import time
import hashlib
import math
from analytics import profile_dataframe, profile_facts, business_facts, compare_periods, save_json
from prompts import SYSTEM, TASKS, REPAIR


class ModelError(RuntimeError):
    pass


def response_schema(facts, count):
    fields = {key: {'type': 'string', 'minLength': 1, 'maxLength': 500}
              for key in ['title', 'observation', 'hypothesis', 'check']}
    fields['evidence_ids'] = {'type': 'array', 'minItems': 1, 'maxItems': 3,
                             'uniqueItems': True,
                             'items': {'type': 'string', 'enum': list(facts)}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['items'],
            'properties': {'items': {'type': 'array', 'minItems': count, 'maxItems': count,
                                    'items': {'type': 'object', 'additionalProperties': False,
                                              'required': list(fields), 'properties': fields}}}}


def validate_response(response, facts, count):
    if not isinstance(response, dict) or set(response) != {'items'}:
        raise ValueError('Очікується JSON-об’єкт лише з ключем items.')
    items = response['items']
    if not isinstance(items, list) or len(items) != count:
        raise ValueError(f'Потрібно рівно {count} пунктів.')
    titles = set()
    for item in items:
        expected = {'title', 'observation', 'hypothesis', 'check', 'evidence_ids'}
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError('Неправильні поля пункту відповіді.')
        for field in expected - {'evidence_ids'}:
            if not isinstance(item[field], str) or not 1 <= len(item[field].strip()) <= 500:
                raise ValueError(f'Некоректний текст поля {field}.')
        refs = item['evidence_ids']
        if not isinstance(refs, list) or not 1 <= len(refs) <= 3:
            raise ValueError('Потрібно від одного до трьох evidence_ids.')
        if any(not isinstance(ref, str) or ref not in facts for ref in refs):
            raise ValueError('Є посилання на неіснуючий факт.')
        if len(set(refs)) != len(refs):
            raise ValueError('Повторні evidence_ids.')
        allowed_numbers = []
        for ref in refs:
            value = facts[ref]['value']
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                allowed_numbers.extend([float(value), round(float(value), 2)])
        for field in expected - {'evidence_ids'}:
            text_to_check = item[field]
            # Номер типу покупки — категорія, а не сума чи обчислена метрика.
            for ref in refs:
                category = re.search(r'feature_type_id=(\d+);', facts[ref].get('label', ''))
                if category:
                    code = re.escape(category.group(1))
                    text_to_check = re.sub(r'(?:типу|тип|типом|feature_type_id)\s*[=:]?\s*' + code + r'\b',
                                           '', text_to_check, flags=re.IGNORECASE)
            known_context = json.dumps(facts, ensure_ascii=False)
            for date in re.findall(r'\d{4}-\d{2}-\d{2}', text_to_check):
                if date not in known_context:
                    raise ValueError(f'Дата {date} відсутня у вхідному контексті.')
                text_to_check = text_to_check.replace(date, '')
            for token in re.findall(r'(?<![\w])[-+]?\d+(?:[.,]\d+)?', text_to_check):
                value = float(token.replace(',', '.'))
                if not any(math.isclose(abs(value), abs(n), rel_tol=0, abs_tol=.00001) for n in allowed_numbers):
                    raise ValueError(f'Число {token} у полі {field} не підтверджене evidence_ids. Прибери його або вкажи правильне джерело; не обчислюй нових чисел.')
        title = item['title'].strip().casefold()
        if title in titles:
            raise ValueError('Повторний заголовок гіпотези.')
        titles.add(title)
    return response


class OllamaClient:
    def __init__(self, model=None, base_url='http://127.0.0.1:11434', timeout=300,
                 log_dir='results/llm_runs'):
        self.model = model or os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.log_dir = Path(log_dir)

    def _request(self, payload):
        request = Request(self.base_url+'/api/chat',
                          data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')[:500]
            raise ModelError(f'Ollama HTTP {exc.code}: {detail}. Перевір ollama list.') from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ModelError('Немає відповіді Ollama. Запусти Ollama, перевір модель і тайм-аут. '
                             'Для локального запуску API-ключ не потрібен.') from exc
        return result

    def analyze(self, task, facts):
        if task not in TASKS or not facts:
            raise ValueError('Невідома задача або порожній набір фактів.')
        count = 5 if task == 'hypotheses' else 3
        schema = response_schema(facts, count)
        user = TASKS[task] + '\nFACTS:\n' + json.dumps(facts, ensure_ascii=False)
        user += '\nПоверни JSON з ключем items. Поля пункту: title, observation, hypothesis, check, evidence_ids. Кількість пунктів: ' + str(count)
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
        payload = {'model': self.model, 'messages': messages, 'stream': False,
                   'format': schema, 'options': {'temperature': 0.1, 'seed': 42, 'repeat_penalty':1.1,
                                                'num_ctx': 8192, 'num_predict': 2400},
                   'keep_alive': '5m'}
        if self.model.startswith('qwen3:'):
            payload['think'] = False
        attempts = []
        started = time.monotonic()
        record = {'task': task, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(),
                  'facts_sha256': hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest(),
                  'facts': facts, 'system_prompt': SYSTEM, 'user_prompt': user, 'attempts': attempts}
        filename = self.log_dir / (datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+task+'.json')
        try:
            for attempt in range(2):
                raw = self._request(payload)
                content = raw.get('message', {}).get('content', '')
                entry = {'content': content, 'eval_count': raw.get('eval_count'),
                         'prompt_eval_count': raw.get('prompt_eval_count'), 'done_reason': raw.get('done_reason')}
                attempts.append(entry)
                if not raw.get('done') or raw.get('done_reason') == 'length':
                    raise ModelError('Модель не завершила відповідь. Збільш num_predict або скороти контекст.')
                try:
                    result = validate_response(json.loads(content), facts, count)
                except (ValueError, TypeError) as exc:
                    entry['validation_error'] = str(exc)
                    if attempt == 1:
                        raise ModelError(f'Відповідь моделі двічі не пройшла перевірку: {exc}') from exc
                    messages.extend([{'role': 'assistant', 'content': content},
                                     {'role': 'user', 'content': REPAIR+str(exc)}])
                    continue
                record.update(status='validated', response=result)
                return {'status': 'live_validated', 'model': self.model, 'task': task,
                        'facts': facts, 'items': result['items'], 'log_file': str(filename),
                        'validation_note': 'Перевірено структуру й джерела чисел. Зміст і причинність оцінює аналітик.'}
        except Exception as exc:
            record.update(status='failed', error=str(exc))
            raise
        finally:
            record['seconds'] = round(time.monotonic()-started, 3)
            save_json(record, filename)


class SavedRunClient:
    """Відтворює лише справжній успішний запуск із тими самими фактами і промптом."""
    def __init__(self, model='qwen2.5:7b', log_dir='results/llm_runs'):
        self.model, self.log_dir = model, Path(log_dir)

    def analyze(self, task, facts):
        fingerprint = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
        for path in sorted(self.log_dir.glob(f'*_{task}.json'), reverse=True):
            record = json.loads(path.read_text(encoding='utf-8'))
            if (record.get('status') == 'validated' and record.get('model') == self.model
                    and record.get('facts_sha256') == fingerprint
                    and record.get('system_prompt') == SYSTEM
                    and record.get('user_prompt', '').startswith(TASKS[task]+'\nFACTS:\n')):
                response = validate_response(record['response'], facts, 5 if task == 'hypotheses' else 3)
                return {'status':'saved_validated', 'model':self.model, 'task':task, 'facts':facts,
                        'items':response['items'], 'log_file':str(path), 'original_timestamp':record['timestamp'],
                        'validation_note':'Збережена справжня відповідь Ollama. Нового API-запиту не було; зміст потребує оцінки аналітика.'}
        raise ModelError(f'Немає підтвердженого запису для {task} з поточними фактами та промптом. Виконай режим live.')


def _analyze(client, task, facts):
    if client is None:
        return {'status': 'not_run', 'task': task, 'facts': facts, 'items': [],
                'message': 'LLM не викликалася. Нижче лише результати Pandas. Для AI-аналізу передай OllamaClient.'}
    return client.analyze(task, facts)


def auto_eda(csv_or_df, client=None, id_columns=('user_id', 'feature_type_id')):
    from analytics import read_csv
    df = read_csv(csv_or_df) if isinstance(csv_or_df, (str, Path)) else csv_or_df
    profile = profile_dataframe(df, id_columns=id_columns)
    return {'profile': profile, 'analysis': _analyze(client, 'eda', profile_facts(profile))}


def suggest_hypotheses(payments, comparison, client=None):
    return _analyze(client, 'hypotheses', business_facts(payments, comparison))


def explain_metric_change(payments, before, after, metric='revenue_usd', client=None):
    comparison = compare_periods(payments, before, after, metric)
    return {'comparison': comparison,
            'analysis': _analyze(client, 'anomaly', business_facts(payments, comparison))}


def executive_summary(payments, comparison, client=None):
    return _analyze(client, 'summary', business_facts(payments, comparison))


def format_analysis(result):
    if result['status'] == 'not_run':
        return result['message']
    lines = [f"Модель: {result['model']} | {result['status']}"]
    for n, item in enumerate(result['items'], 1):
        lines += [f"\n{n}. {item['title']}", 'Спостереження: '+item['observation'],
                  'Гіпотеза / обмеження: '+item['hypothesis'], 'Перевірка: '+item['check']]
        for ref in item['evidence_ids']:
            fact = result['facts'][ref]
            value = fact['value']
            formatted = f'{value:,.2f}' if isinstance(value, float) else str(value)
            lines.append(f"  [{ref}] {fact['label']}: {formatted} {fact['unit']}")
    lines.append('\n'+result['validation_note'])
    return '\n'.join(lines)
