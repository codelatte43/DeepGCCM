from __future__ import annotations
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from baseline.data_loader import CAUSAL_DATASETS, load_baseline_datasets
from baseline.methods import ALL_METHODS, run_all_baselines

def parse_args():
    p = argparse.ArgumentParser(description='Run baseline causal-discovery methods.')
    p.add_argument('--data-dir', default='./datasets')
    p.add_argument('--cache-dir', default='./data_cache')
    p.add_argument('--results-dir', default='./baseline/results')
    p.add_argument('--subsample', type=int, default=20000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quick', action='store_true', help='Use 2000 cells for a fast smoke test.')
    p.add_argument('--methods', default=','.join(ALL_METHODS.keys()), help='Comma-separated method names.')
    return p.parse_args()

def results_to_dataframe(results) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({'method': r.method, 'dataset_id': r.dataset_id, 'score_xy': round(r.score_xy, 4) if r.score_xy == r.score_xy else None, 'score_yx': round(r.score_yx, 4) if r.score_yx == r.score_yx else None, 'predicted_direction': r.predicted_direction, 'direction_correct': r.direction_correct, 'extra': str(r.extra) if r.extra else ''})
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    if args.quick:
        args.subsample = 2000
    os.makedirs(args.results_dir, exist_ok=True)
    method_list = [m.strip() for m in args.methods.split(',') if m.strip()]
    print('=' * 70)
    print('Baseline Causal Discovery — USA Human–Environment (3 datasets)')
    print('=' * 70)
    print(f'Methods: {method_list}')
    print(f'Subsample: {args.subsample}')
    optional = []
    for (pkg, label) in [('lingam', 'LiNGAM'), ('tigramite', 'PCMCI+')]:
        try:
            __import__(pkg)
        except ImportError:
            optional.append(label)
    if optional:
        print(f"Note: install optional deps for {', '.join(optional)}:\n  pip install -r baseline/requirements.txt")
    datasets = load_baseline_datasets(data_dir=args.data_dir, cache_dir=args.cache_dir, subsample=args.subsample, seed=args.seed)
    all_results = []
    all_failures = []
    for ds in datasets:
        print(f'\n--- {ds.name}  (n={ds.n_spatial:,}) ---')
        (res, failures) = run_all_baselines(ds, methods=method_list)
        all_results.extend(res)
        all_failures.extend(failures)
        for r in res:
            if np.isfinite(r.score_xy):
                sxy = f'{r.score_xy:+.4f}'
            else:
                sxy = '   nan'
            if np.isfinite(r.score_yx):
                syx = f'{r.score_yx:+.4f}'
            else:
                syx = '   nan'
            if r.predicted_direction in {'undirected', 'unavailable'}:
                mark = '—'
            elif r.direction_correct:
                mark = 'OK'
            else:
                mark = 'X'
            print(f'  {r.method:<20s}  score_xy={sxy}  score_yx={syx}  pred={r.predicted_direction:<12s} [{mark}]')
    if all_failures:
        print('\nWarnings:')
        for msg in all_failures:
            print(f'  - {msg}')
    df = results_to_dataframe(all_results)
    out_path = os.path.join(args.results_dir, 'baseline_table.csv')
    df.to_csv(out_path, index=False)
    directed = df[~df['predicted_direction'].isin(['undirected', 'unavailable'])].copy()
    summary = directed.groupby('method')['direction_correct'].agg(n='count', accuracy='mean').reset_index()
    summary['accuracy'] = (summary['accuracy'] * 100).round(1)
    summary_path = os.path.join(args.results_dir, 'baseline_summary.csv')
    summary.to_csv(summary_path, index=False)
    print('\n' + '=' * 70)
    print('SUMMARY (direction accuracy, directed methods only)')
    print('=' * 70)
    print(summary.to_string(index=False))
    print(f'\nSaved: {os.path.abspath(out_path)}')
    print(f'Saved: {os.path.abspath(summary_path)}')
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
