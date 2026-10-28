import unittest
import copy
import tempfile
from pathlib import Path
import pandas as pd
from analytics import compare_periods, profile_dataframe, read_csv, load_data, metrics
from assistant import validate_response, OllamaClient, ModelError, auto_eda, SavedRunClient


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'user_id': [1,2,3,1,4],
            'pay_dt': pd.to_datetime(['2024-01-01','2024-01-02','2024-01-03','2024-01-08','2024-01-10']),
            'amount_cents': [1000,2000,3000,500,1000],
            'app': ['ios','ios','android','ios','new'],
            'country_code': ['UA','DE','UA','UA','PL'],
            'feature_type_id': [1,1,2,1,2]})

    def test_empty_eda(self):
        with self.assertRaises(ValueError):
            profile_dataframe(pd.DataFrame())

    def test_all_missing_column(self):
        profile = profile_dataframe(pd.DataFrame({'x': [float('nan')]*3}))
        self.assertEqual(profile['columns'][0]['missing'], 3)
        self.assertEqual(profile['distributions']['x']['finite_count'], 0)

    def test_infinity_flag(self):
        profile = profile_dataframe(pd.DataFrame({'x': [1.,2.,float('inf')]}))
        self.assertEqual(profile['distributions']['x']['nonfinite'], 1)

    def test_na_country_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample.csv'
            path.write_text('country_code,age\nNA,\nUA,20\n', encoding='utf-8')
            df = read_csv(path)
            self.assertEqual(df.loc[0, 'country_code'], 'NA')
            self.assertTrue(pd.isna(df.loc[0, 'age']))

    def test_contributions_include_new_and_lost_segments(self):
        result = compare_periods(self.df, ('2024-01-01','2024-01-03'), ('2024-01-08','2024-01-10'))
        self.assertEqual(result['delta'], -45)
        table = result['breakdown']['app'].set_index('segment')
        self.assertEqual(table.loc['new','delta'], 10)
        self.assertEqual(table.loc['android','delta'], -30)
        self.assertEqual(table.delta.sum(), result['delta'])

    def test_zero_baseline(self):
        self.df.loc[self.df.pay_dt < '2024-01-08', 'amount_cents'] = 0
        result = compare_periods(self.df, ('2024-01-01','2024-01-03'), ('2024-01-08','2024-01-10'))
        self.assertIsNone(result['change_pct'])

    def test_zero_delta(self):
        self.df.loc[self.df.pay_dt == '2024-01-08', 'amount_cents'] = 5000
        result = compare_periods(self.df, ('2024-01-01','2024-01-03'), ('2024-01-08','2024-01-10'))
        self.assertTrue(result['breakdown']['app'].contribution_pct.isna().all())

    def test_count_metric(self):
        result = compare_periods(self.df, ('2024-01-01','2024-01-03'), ('2024-01-08','2024-01-10'), metric='payments')
        self.assertEqual(result['delta'], -1)

    def test_bad_periods(self):
        cases = [
            (('2024-01-01','2024-01-02'), ('2024-01-08','2024-01-10')),
            (('2024-01-01','2024-01-03'), ('2024-01-03','2024-01-05')),
            (('2023-12-01','2023-12-03'), ('2024-01-08','2024-01-10')),
        ]
        for before, after in cases:
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                compare_periods(self.df, before, after)

    def test_unsupported_metric(self):
        with self.assertRaises(ValueError):
            compare_periods(self.df, metric='retention')

    def test_no_model_is_not_ai(self):
        result = auto_eda(pd.DataFrame({'amount_usd':[1.,2.,3.]}))
        self.assertEqual(result['analysis']['status'], 'not_run')
        self.assertEqual(result['analysis']['items'], [])


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.facts = {'F001': {'label':'Виручка', 'value':42, 'unit':'USD'}}
        self.valid = {'items':[{'title':'Зміна виручки', 'observation':'Виручка змінилася.',
                               'hypothesis':'Можлива зміна частоти покупок.',
                               'check':'Порівняти кількість покупок.', 'evidence_ids':['F001']}]}

    def test_valid_response(self):
        self.assertIs(validate_response(self.valid, self.facts, 1), self.valid)

    def test_unknown_reference(self):
        self.valid['items'][0]['evidence_ids'] = ['F999']
        with self.assertRaises(ValueError):
            validate_response(self.valid, self.facts, 1)

    def test_invented_number(self):
        self.valid['items'][0]['observation'] = 'Виручка дорівнює 999.'
        with self.assertRaises(ValueError):
            validate_response(self.valid, self.facts, 1)

    def test_supported_number(self):
        self.valid['items'][0]['observation'] = 'Виручка становить 42 USD.'
        validate_response(self.valid, self.facts, 1)

    def test_category_identifier_is_not_a_metric(self):
        self.facts['F001']['label'] = 'feature_type_id=1; revenue_usd: before'
        self.valid['items'][0]['observation'] = 'Виручка типу 1 змінилася.'
        validate_response(self.valid, self.facts, 1)
        self.valid['items'][0]['observation'] = 'Виручка становить 1 USD.'
        with self.assertRaises(ValueError):
            validate_response(self.valid, self.facts, 1)
        self.valid['items'][0]['observation'] = 'Виручка типу 2 змінилася.'
        with self.assertRaises(ValueError):
            validate_response(self.valid, self.facts, 1)

    def test_unsupported_date(self):
        self.valid['items'][0]['observation'] = 'Падіння сталося 2030-01-01.'
        with self.assertRaises(ValueError):
            validate_response(self.valid, self.facts, 1)

    def test_no_saved_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ModelError):
                SavedRunClient(log_dir=tmp).analyze('summary', self.facts)

    def test_missing_field(self):
        del self.valid['items'][0]['check']
        with self.assertRaises(ValueError):
            validate_response(self.valid, self.facts, 1)

    def test_repair_and_log(self):
        import json
        items = []
        for name in ['Перший','Другий','Третій']:
            item = copy.deepcopy(self.valid['items'][0]); item['title'] = name; items.append(item)
        responses = iter(['not json', json.dumps({'items':items}, ensure_ascii=False)])
        with tempfile.TemporaryDirectory() as tmp:
            client = OllamaClient(log_dir=tmp)
            client._request = lambda payload: {'done':True, 'message':{'content':next(responses)}}
            result = client.analyze('summary', self.facts)
            self.assertEqual(result['status'], 'live_validated')
            log = json.loads(next(Path(tmp).glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(len(log['attempts']), 2)
            self.assertIn('validation_error', log['attempts'][0])

    def test_two_bad_responses_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = OllamaClient(log_dir=tmp)
            client._request = lambda payload: {'done':True, 'message':{'content':'{}'}}
            with self.assertRaises(ModelError):
                client.analyze('summary', self.facts)


class DatasetTests(unittest.TestCase):
    def test_real_dataset_control(self):
        root = Path(__file__).resolve().parents[1]
        users, payments, rejected, audit = load_data(root/'data')
        result = metrics(payments)
        self.assertEqual(len(users), 202382)
        self.assertEqual(len(payments), 11330)
        self.assertEqual(result['revenue_usd'], 92240.40)
        self.assertEqual(result['payments'], 11159)
        self.assertEqual(result['paying_users'], 7664)
        # Незалежний контроль без злиття довідників, через Decimal.
        from decimal import Decimal
        import csv
        total = Decimal('0')
        with (root/'data/payments.csv').open(encoding='utf-8') as file:
            for row in csv.DictReader(file):
                if '2024-04-05' <= row['pay_dt'][:10] <= '2024-05-10':
                    total += Decimal(row['amount_usd'])
        self.assertEqual(total, Decimal(str(result['revenue_usd'])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
