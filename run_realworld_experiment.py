from __future__ import annotations
import argparse
import os
import time
import tracemalloc
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from src.data import USA_CAUSAL_PAIRS, USAGridDataset, add_noise, aggregate_usa_grid, load_usa_human_env, preprocessed_to_grid, _assign_state
from src.deepgccm import TrainConfig, deepgccm_convergence, extract_cross_attention, extract_gat_attention, train_deepgccm
from src.gccm import gccm_convergence, select_E
from src.metrics import cohens_d, direction_predicted, full_metrics, library_size_fractions, mean_ci, paired_t_test, wilcoxon_signed_rank
from src.plotting import plot_ablation, plot_attention_heatmap, plot_causal_direction_comparison, plot_convergence, plot_convergence_multi, plot_multiscale, plot_neighbor_importance_map, plot_noise_robustness, plot_runtime_scaling, plot_scatter, plot_spatial_variable, plot_study_area, plot_temporal_transfer
from src.preprocess_usa import PreprocessConfig, run_preprocessing
ALL_DIRECTIONS = [('Population', 'Nightlight', 'Population->Nightlight'), ('Nightlight', 'Population', 'Nightlight->Population'), ('Nightlight', 'PM2.5', 'Nightlight->PM2.5'), ('PM2.5', 'Nightlight', 'PM2.5->Nightlight'), ('Population', 'PM2.5', 'Population->PM2.5'), ('PM2.5', 'Population', 'PM2.5->Population')]
FORWARD_DIRECTIONS = [('Population', 'Nightlight', 'Population->Nightlight'), ('Nightlight', 'PM2.5', 'Nightlight->PM2.5'), ('Population', 'PM2.5', 'Population->PM2.5')]
SCALES_KM = [1, 5, 10, 25, 50]
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
SCALABILITY_SIZES = [1000, 5000, 10000, 20000, 50000, 100000]

def parse_args():
    p = argparse.ArgumentParser(description='Run USA human-environment experiments.')
    p.add_argument('--data-dir', default='./datasets')
    p.add_argument('--cache-dir', default='./data_cache')
    p.add_argument('--results-dir', default='./results/realworld')
    p.add_argument('--figures-dir', default='./figures')
    p.add_argument('--subsample', type=int, default=8000, help='Max spatial cells for main experiments.')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--n-repeat', type=int, default=15)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quick', action='store_true', help='Smoke test: 60 epochs, 3 repeats, 1500 cells.')
    p.add_argument('--skip-preprocess', action='store_true')
    p.add_argument('--experiments', default='1,2,3,4,5,6,7,8', help='Comma-separated experiment IDs to run.')
    return p.parse_args()

def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else a - a.mean()

def _subsample_grid(grid: USAGridDataset, n: int, seed: int) -> USAGridDataset:
    if grid.n <= n:
        return grid
    rng = np.random.default_rng(seed)
    idx = rng.choice(grid.n, size=n, replace=False)
    return USAGridDataset(name=f'{grid.name} (n={n})', coords=grid.coords[idx], lonlat=grid.lonlat[idx], cell_id=grid.cell_id[idx], population=grid.population[idx], nightlight=grid.nightlight[idx], pm25=grid.pm25[idx], cell_size_km=grid.cell_size_km, source=grid.source)

def _pair_dataset(grid: USAGridDataset, x_name: str, y_name: str, label: str):
    td = 'x->y' if (x_name, y_name) in [(a, b) for (a, b, _) in USA_CAUSAL_PAIRS] else 'y->x'
    ds = grid.to_pair(x_name, y_name, true_direction=td)
    ds.name = label
    return ds

def _run_pair(ds, args, lib_sizes=None, train_overrides: Optional[dict]=None, x_override: Optional[np.ndarray]=None, y_override: Optional[np.ndarray]=None) -> Tuple[dict, dict, object, TrainConfig, int]:
    x = x_override if x_override is not None else ds.x
    y = y_override if y_override is not None else ds.y
    E = select_E(x, y, ds.coords, E_candidates=(2, 3, 4, 5))
    gccm_res = gccm_convergence(x, y, ds.coords, E=E, lib_sizes=lib_sizes, n_repeat=args.n_repeat, seed=args.seed)
    cfg = TrainConfig(epochs=args.epochs, device=args.device, seed=args.seed)
    if train_overrides:
        for (k, v) in train_overrides.items():
            setattr(cfg, k, v)
    (model, log) = train_deepgccm(x, y, ds.coords, cfg)
    deep_res = deepgccm_convergence(model, x, y, ds.coords, k_neighbors=cfg.k_neighbors, lib_sizes=lib_sizes, n_repeat=args.n_repeat, seed=args.seed, train_elapsed_sec=log['elapsed_sec'])
    return (gccm_res, deep_res, model, cfg, E)

def _row_table1(direction: str, gccm_res: dict, deep_res: dict, y_actual: np.ndarray) -> dict:
    (g_pred, d_pred) = (gccm_res['y_hat_full'], deep_res['y_hat_full'])
    g_m = full_metrics(g_pred, y_actual)
    d_m = full_metrics(d_pred, y_actual)
    return {'direction': direction, 'rho_gccm': round(g_m['rho'], 4), 'rho_deepgccm': round(d_m['rho'], 4), 'rmse_gccm': round(g_m['rmse'], 4), 'rmse_deepgccm': round(d_m['rmse'], 4), 'mae_gccm': round(g_m['mae'], 4), 'mae_deepgccm': round(d_m['mae'], 4), 'r2_gccm': round(g_m['r2'], 4), 'r2_deepgccm': round(d_m['r2'], 4), 'rho_improvement': round(d_m['rho'] - g_m['rho'], 4), 'rmse_improvement_pct': round(100 * (g_m['rmse'] - d_m['rmse']) / (g_m['rmse'] + 1e-12), 2)}

def experiment_1(grid: USAGridDataset, args) -> pd.DataFrame:
    print('\n' + '=' * 70)
    print('EXPERIMENT 1 — Real-world spatial causal discovery')
    print('=' * 70)
    rows = []
    direction_rows = []
    for (x_name, y_name, label) in ALL_DIRECTIONS:
        ds = _pair_dataset(grid, x_name, y_name, label)
        print(f'  {label}  (n={ds.n})')
        (gccm_res, deep_res, _, _, E) = _run_pair(ds, args)
        y_actual = ds.y
        rows.append(_row_table1(label, gccm_res, deep_res, y_actual))
        direction_rows.append({'direction': label, 'rho_gccm': gccm_res['rho_xmap_y_full'], 'rho_deep': deep_res['rho_xmap_y_full'], 'predicted_gccm': direction_predicted(gccm_res['rho_xmap_y_full'], gccm_res['rho_ymap_x_full']), 'predicted_deep': direction_predicted(deep_res['rho_xmap_y_full'], deep_res['rho_ymap_x_full'])})
        plot_convergence(ds, gccm_res, deep_res, os.path.join(args.figures_dir, f"convergence_{label.replace('->', '_to_')}.png"))
        plot_scatter(ds, gccm_res, deep_res, os.path.join(args.figures_dir, f"scatter_{label.replace('->', '_to_')}.png"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, 'table1_causal_discovery.csv'), index=False)
    plot_causal_direction_comparison(direction_rows, os.path.join(args.figures_dir, 'fig_causal_direction_comparison.png'))
    print(df.to_string(index=False))
    return df

def experiment_2(grid: USAGridDataset, args) -> pd.DataFrame:
    print('\n' + '=' * 70)
    print('EXPERIMENT 2 — Convergence analysis')
    print('=' * 70)
    lib_sizes = library_size_fractions(grid.n)
    rows = []
    curves_plot = []
    for (x_name, y_name, label) in FORWARD_DIRECTIONS:
        ds = _pair_dataset(grid, x_name, y_name, label)
        (gccm_res, deep_res, _, _, _) = _run_pair(ds, args, lib_sizes=lib_sizes)
        for (method, res, key) in [('GCCM', gccm_res, 'rho_xmap_y_curve'), ('DeepGCCM', deep_res, 'rho_xmap_y_curve')]:
            curve = res[key]
            for (L, rho_m, rho_s) in zip(curve.lib_sizes, curve.rho_mean, curve.rho_std):
                rows.append({'direction': label, 'method': method, 'lib_size': int(L), 'lib_frac': L / grid.n, 'rho': rho_m, 'rho_std': rho_s})
            curves_plot.append({'label': label, 'method': method, 'lib_sizes': curve.lib_sizes, 'rho_mean': curve.rho_mean, 'rho_std': curve.rho_std})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, 'convergence_table.csv'), index=False)
    plot_convergence_multi(curves_plot, os.path.join(args.figures_dir, 'fig05_convergence_curves.png'), title='Figure 5 — Convergence curves (forward causal pairs)')
    return df

def experiment_3(args) -> pd.DataFrame:
    print('\n' + '=' * 70)
    print('EXPERIMENT 3 — Multi-scale spatial causality')
    print('=' * 70)
    rows = []
    base_grid = load_usa_human_env(data_dir=args.data_dir, cache_dir=args.cache_dir, subsample=None, split='2023')
    for scale in SCALES_KM:
        if scale == 1:
            g = _subsample_grid(base_grid, args.subsample, args.seed)
        else:
            g = aggregate_usa_grid(base_grid, cell_size_km=scale)
            g = _subsample_grid(g, min(args.subsample, g.n), args.seed)
        print(f'  scale={scale} km, n={g.n}')
        for (x_name, y_name, label) in FORWARD_DIRECTIONS:
            ds = _pair_dataset(g, x_name, y_name, label)
            (gccm_res, deep_res, _, _, _) = _run_pair(ds, args, train_overrides={'epochs': max(args.epochs // 2, 80)})
            rows.append({'scale_km': scale, 'direction': label, 'method': 'GCCM', 'rho': gccm_res['rho_xmap_y_full'], 'n': g.n})
            rows.append({'scale_km': scale, 'direction': label, 'method': 'DeepGCCM', 'rho': deep_res['rho_xmap_y_full'], 'n': g.n})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, 'multiscale_table.csv'), index=False)
    plot_multiscale(rows, os.path.join(args.figures_dir, 'fig07_multiscale_analysis.png'))
    return df

def experiment_4(args) -> pd.DataFrame:
    print('\n' + '=' * 70)
    print('EXPERIMENT 4 — Temporal generalization')
    print('=' * 70)
    train_grid = load_usa_human_env(data_dir=args.data_dir, cache_dir=args.cache_dir, subsample=args.subsample, split='train')
    test_grid = load_usa_human_env(data_dir=args.data_dir, cache_dir=args.cache_dir, subsample=args.subsample, split='test', seed=args.seed + 1)
    rows = []
    for (x_name, y_name, label) in FORWARD_DIRECTIONS:
        train_ds = _pair_dataset(train_grid, x_name, y_name, label)
        test_ds = _pair_dataset(test_grid, x_name, y_name, label)
        cfg = TrainConfig(epochs=args.epochs, device=args.device, seed=args.seed)
        (model, _) = train_deepgccm(train_ds.x, train_ds.y, train_ds.coords, cfg)
        E = select_E(test_ds.x, test_ds.y, test_ds.coords)
        gccm_test = gccm_convergence(test_ds.x, test_ds.y, test_ds.coords, E=E, n_repeat=args.n_repeat, seed=args.seed)
        deep_test = deepgccm_convergence(model, test_ds.x, test_ds.y, test_ds.coords, k_neighbors=cfg.k_neighbors, n_repeat=args.n_repeat, seed=args.seed)
        for (method, res) in [('GCCM', gccm_test), ('DeepGCCM', deep_test)]:
            m = full_metrics(res['y_hat_full'], test_ds.y)
            rows.append({'direction': label, 'method': method, 'split': 'test_2024_2025', **{k: round(v, 4) for (k, v) in m.items()}})
        print(f"  {label}: GCCM rho={gccm_test['rho_xmap_y_full']:+.3f}, Deep rho={deep_test['rho_xmap_y_full']:+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, 'temporal_transfer_table.csv'), index=False)
    plot_temporal_transfer(rows, os.path.join(args.figures_dir, 'fig_temporal_transfer.png'))
    return df

def experiment_5(grid: USAGridDataset, args) -> pd.DataFrame:
    print('\n' + '=' * 70)
    print('EXPERIMENT 5 — Noise robustness')
    print('=' * 70)
    rows = []
    for level in NOISE_LEVELS:
        for (x_name, y_name, label) in FORWARD_DIRECTIONS:
            ds = _pair_dataset(grid, x_name, y_name, label)
            x_n = add_noise(ds.x, level, seed=args.seed)
            y_n = add_noise(ds.y, level, seed=args.seed + 1)
            overrides = {'epochs': max(args.epochs // 2, 80)}
            (gccm_res, deep_res, _, _, _) = _run_pair(ds, args, train_overrides=overrides, x_override=x_n, y_override=y_n)
            for (method, res) in [('Traditional GCCM', gccm_res), ('DeepGCCM', deep_res)]:
                rows.append({'direction': label, 'method': method, 'noise': level, 'rho': res['rho_xmap_y_full'], 'rmse': res['rmse_xmap_y_full']})
        print(f'  noise={level:.0%} done')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, 'noise_table.csv'), index=False)
    plot_noise_robustness(rows, os.path.join(args.figures_dir, 'fig08_noise_robustness.png'))
    return df

def experiment_6(grid: USAGridDataset, args) -> None:
    print('\n' + '=' * 70)
    print('EXPERIMENT 6 — Spatial neighbour learning (attention)')
    print('=' * 70)
    ds = _pair_dataset(grid, 'Population', 'Nightlight', 'Population->Nightlight')
    cfg = TrainConfig(epochs=args.epochs, device=args.device, seed=args.seed)
    (model, _) = train_deepgccm(ds.x, ds.y, ds.coords, cfg)
    attn = extract_gat_attention(model, ds.x, ds.y, ds.coords, k_neighbors=cfg.k_neighbors, variable='x')
    plot_neighbor_importance_map(grid, attn['alpha'], os.path.join(args.figures_dir, 'fig09_neighbor_importance_map.png'))
    urban_idx = np.argsort(grid.population)[-20:]
    plot_attention_heatmap(attn['alpha'], attn['nbr_idx'], urban_idx, os.path.join(args.figures_dir, 'fig09_attention_heatmap_urban.png'), title='GAT attention — top 20 urban cells')
    cross = extract_cross_attention(model, ds.x, ds.y, ds.coords, k_neighbors=cfg.k_neighbors, direction='xmap_y')
    pd.DataFrame({'mean_library_attention': cross['mean_alpha']}).to_csv(os.path.join(args.results_dir, 'attention_library_weights.csv'), index=False)
    print('  Attention maps saved to figures/')

def _state_group_ids(lonlat: np.ndarray) -> np.ndarray:
    names = [_assign_state(float(lon), float(lat)) for (lon, lat) in lonlat]
    uniq = {n: i for (i, n) in enumerate(sorted(set(names)))}
    return np.asarray([uniq[n] for n in names], dtype=np.int64)

def experiment_7(grid: USAGridDataset, args) -> pd.DataFrame:
    print('\n' + '=' * 70)
    print('EXPERIMENT 7 — Ablation study (H1/H2/H3, fair protocol)')
    print('=' * 70)
    rows = []
    ablate_epochs = args.epochs
    fair = dict(geo_exclude_k=16, attn_topk=8)
    variants = {'A: Traditional GCCM': None, 'B: DeepGCCM w/o GNN': dict(use_gnn=False, use_attention=True, **fair), 'C: DeepGCCM w/o Attention': dict(use_gnn=True, use_attention=False, **fair), 'D: Full DeepGCCM': dict(use_gnn=True, use_attention=True, **fair)}
    plot_key = {'B: DeepGCCM w/o GNN': 'no_gnn', 'C: DeepGCCM w/o Attention': 'no_attn', 'D: Full DeepGCCM': 'full'}
    task_label = {'Population->Nightlight': 'H1 Pop→NL', 'Nightlight->PM2.5': 'H2 NL→PM', 'Population->PM2.5': 'H3 Pop→PM'}
    state_ids = _state_group_ids(grid.lonlat)
    for (x_name, y_name, direction) in FORWARD_DIRECTIONS:
        ds = _pair_dataset(grid, x_name, y_name, direction)
        short = task_label.get(direction, direction)
        use_state_ban = 'PM2.5' in (x_name, y_name)
        group_id = state_ids if use_state_ban else None
        print(f"\n  --- {short} ({direction}, n={ds.n}){(' [same-state ban]' if use_state_ban else '')} ---")
        for (name, flags) in variants.items():
            if flags is None:
                E = select_E(ds.x, ds.y, ds.coords)
                res = gccm_convergence(ds.x, ds.y, ds.coords, E=E, n_repeat=args.n_repeat, seed=args.seed)
                row = {'direction': direction, 'task': short, 'variant': name, 'rho': res['rho_xmap_y_full'], 'rmse': res['rmse_xmap_y_full'], 'time_sec': res['elapsed_sec'], 'geo_exclude_k': 0, 'attn_topk': None, 'same_state_ban': False}
            else:
                geo_k = flags.get('geo_exclude_k', 0)
                cfg = TrainConfig(epochs=ablate_epochs, device=args.device, seed=args.seed, **flags)
                (model, log) = train_deepgccm(ds.x, ds.y, ds.coords, cfg, group_id=group_id)
                res = deepgccm_convergence(model, ds.x, ds.y, ds.coords, k_neighbors=cfg.k_neighbors, n_repeat=args.n_repeat, seed=args.seed, train_elapsed_sec=log['elapsed_sec'], geo_exclude_k=geo_k, group_id=group_id)
                row = {'direction': direction, 'task': short, 'variant': name, 'rho': res['rho_xmap_y_full'], 'rmse': res['rmse_xmap_y_full'], 'time_sec': log['elapsed_sec'] + res['elapsed_sec'], 'geo_exclude_k': geo_k, 'attn_topk': flags.get('attn_topk'), 'same_state_ban': bool(use_state_ban)}
            rows.append(row)
            print(f"    {name}: rho={row['rho']:+.3f}  rmse={row['rmse']:.3f}")
    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.results_dir, 'ablation_table.csv')
    df.to_csv(out_csv, index=False)
    summary = df.groupby('variant', sort=False).agg(rho_mean=('rho', 'mean'), rho_std=('rho', 'std'), rmse_mean=('rmse', 'mean'), time_mean=('time_sec', 'mean')).reset_index()
    summary.to_csv(os.path.join(args.results_dir, 'ablation_summary_H1H2H3.csv'), index=False)
    print('\n  === Mean over H1–H3 (fair ablation protocol) ===')
    print(summary.to_string(index=False))
    print('  Protocol: DeepGCCM uses geo_exclude_k=16, attn_topk=8;')
    print('            H2/H3 additionally ban same-state library cells.')
    print('            GCCM baseline keeps the original protocol.')
    ablation_plot_rows = [{'dataset': r['task'], 'variant': plot_key[r['variant']], 'rho_xy': r['rho'], 'rho_yx': r['rho']} for r in rows if r['variant'] in plot_key]
    plot_ablation(ablation_plot_rows, os.path.join(args.figures_dir, 'fig_ablation.png'))
    print(f'  Saved: {out_csv}')
    return df

def experiment_8(args, scalability_sizes: Optional[List[int]]=None) -> pd.DataFrame:
    print('\n' + '=' * 70)
    print('EXPERIMENT 8 — Scalability analysis')
    print('=' * 70)
    full_grid = load_usa_human_env(data_dir=args.data_dir, cache_dir=args.cache_dir, subsample=None, split='2023')
    ds_full = _pair_dataset(full_grid, 'Population', 'Nightlight', 'Population->Nightlight')
    rows = []
    sizes = [s for s in scalability_sizes or SCALABILITY_SIZES if s <= full_grid.n]
    for n in sizes:
        g = _subsample_grid(full_grid, n, args.seed)
        ds = _pair_dataset(g, 'Population', 'Nightlight', 'Population->Nightlight')
        tracemalloc.start()
        t0 = time.time()
        gccm_res = gccm_convergence(ds.x, ds.y, ds.coords, E=3, n_repeat=3, seed=args.seed)
        gccm_time = time.time() - t0
        (_, gccm_peak) = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        train_lib = 0.5 if n > 15000 else 0.8
        cfg = TrainConfig(epochs=max(args.epochs // 4, 60), device=args.device, seed=args.seed, train_lib_frac=train_lib)
        try:
            (model, log) = train_deepgccm(ds.x, ds.y, ds.coords, cfg)
            deep_res = deepgccm_convergence(model, ds.x, ds.y, ds.coords, k_neighbors=cfg.k_neighbors, n_repeat=3, seed=args.seed, train_elapsed_sec=log['elapsed_sec'])
            deep_time = log['elapsed_sec'] + deep_res['elapsed_sec']
            deep_rho = deep_res['rho_xmap_y_full']
            deep_mem = tracemalloc.get_traced_memory()[1] / 1000000.0
        except torch.cuda.OutOfMemoryError:
            warnings.warn(f'GPU OOM at n={n}; retrying DeepGCCM on CPU.')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cfg.device = 'cpu'
            (model, log) = train_deepgccm(ds.x, ds.y, ds.coords, cfg)
            deep_res = deepgccm_convergence(model, ds.x, ds.y, ds.coords, k_neighbors=cfg.k_neighbors, n_repeat=3, seed=args.seed, train_elapsed_sec=log['elapsed_sec'])
            deep_time = log['elapsed_sec'] + deep_res['elapsed_sec']
            deep_rho = deep_res['rho_xmap_y_full']
            deep_mem = tracemalloc.get_traced_memory()[1] / 1000000.0
        finally:
            tracemalloc.stop()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append({'n': n, 'method': 'GCCM', 'time_sec': gccm_time, 'memory_mb': gccm_peak / 1000000.0, 'rho': gccm_res['rho_xmap_y_full']})
        rows.append({'n': n, 'method': 'DeepGCCM', 'time_sec': deep_time, 'memory_mb': deep_mem, 'rho': deep_rho})
        print(f'  n={n:6d}  GCCM={gccm_time:.1f}s  Deep={deep_time:.1f}s')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, 'scalability_table.csv'), index=False)
    plot_runtime_scaling(rows, os.path.join(args.figures_dir, 'fig10_runtime_scaling.png'))
    mem_rows = [{'n': r['n'], 'method': r['method'], 'memory_mb': r['memory_mb']} for r in rows]
    plot_runtime_scaling(mem_rows, os.path.join(args.figures_dir, 'fig10_memory_scaling.png'), metric='memory_mb')
    return df

def run_statistics(table1: pd.DataFrame, args) -> pd.DataFrame:
    print('\n' + '=' * 70)
    print('STATISTICAL ANALYSIS')
    print('=' * 70)
    deep = table1['rho_deepgccm'].values
    gccm = table1['rho_gccm'].values
    (t_stat, p_t) = paired_t_test(deep, gccm)
    (w_stat, p_w) = wilcoxon_signed_rank(deep, gccm)
    deltas = deep - gccm
    (mean, lo, hi) = mean_ci(deltas)
    d_effect = cohens_d(deep, gccm)
    stats_rows = []
    for (_, row) in table1.iterrows():
        stats_rows.append({'direction': row['direction'], 'rho_gccm': row['rho_gccm'], 'rho_deepgccm': row['rho_deepgccm'], 'delta': row['rho_improvement']})
    summary = {'direction': 'SUMMARY', 'rho_gccm': round(float(np.mean(gccm)), 4), 'rho_deepgccm': round(float(np.mean(deep)), 4), 'delta': round(float(mean), 4), 'ci_95_lo': round(float(lo), 4), 'ci_95_hi': round(float(hi), 4), 'paired_t': round(t_stat, 4) if np.isfinite(t_stat) else None, 'p_t': round(p_t, 6) if np.isfinite(p_t) else None, 'wilcoxon_W': round(w_stat, 1) if np.isfinite(w_stat) else None, 'p_wilcoxon': round(p_w, 6) if np.isfinite(p_w) else None, 'cohens_d': round(d_effect, 3)}
    stats_rows.append(summary)
    df = pd.DataFrame(stats_rows)
    df.to_csv(os.path.join(args.results_dir, 'stats_tests.csv'), index=False)
    print(df.to_string(index=False))
    return df

def main():
    args = parse_args()
    if args.quick:
        args.epochs = 60
        args.n_repeat = 3
        args.subsample = 1500
        scalability_sizes = [500, 1000, 1500]
    else:
        scalability_sizes = [s for s in SCALABILITY_SIZES]
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    exp_ids = {int(x.strip()) for x in args.experiments.split(',')}
    print('=' * 70)
    print('DeepGCCM Real-world Experiment — USA Human–Environment System')
    print('=' * 70)
    if not args.skip_preprocess:
        print('\nPreprocessing USA datasets ...')
        cfg = PreprocessConfig(data_dir=args.data_dir, cache_dir=args.cache_dir)
        run_preprocessing(cfg, save=True)
    grid = load_usa_human_env(data_dir=args.data_dir, cache_dir=args.cache_dir, subsample=args.subsample, split='2023')
    print(f'Loaded grid: n={grid.n:,}, cell_size={grid.cell_size_km} km, source={grid.source}')
    plot_study_area(grid, os.path.join(args.figures_dir, 'fig01_study_area_datasets.png'))
    plot_spatial_variable(grid, 'population', os.path.join(args.figures_dir, 'fig02_spatial_population.png'))
    plot_spatial_variable(grid, 'nightlight', os.path.join(args.figures_dir, 'fig03_spatial_nightlight.png'))
    plot_spatial_variable(grid, 'pm25', os.path.join(args.figures_dir, 'fig04_spatial_pm25.png'), log_scale=False)
    table1 = None
    if 1 in exp_ids:
        table1 = experiment_1(grid, args)
    if 2 in exp_ids:
        experiment_2(grid, args)
    if 3 in exp_ids:
        experiment_3(args)
    if 4 in exp_ids:
        experiment_4(args)
    if 5 in exp_ids:
        experiment_5(grid, args)
    if 6 in exp_ids:
        experiment_6(grid, args)
    if 7 in exp_ids:
        experiment_7(grid, args)
    if 8 in exp_ids:
        experiment_8(args, scalability_sizes)
    if table1 is not None:
        run_statistics(table1, args)
    conclusion = ['DeepGCCM Real-world Experiment — Summary', '=' * 50, f'Grid: n={grid.n:,}, source={grid.source}', f'Hypotheses: Population->Nightlight->PM2.5', f'Results directory: {os.path.abspath(args.results_dir)}', f'Figures directory: {os.path.abspath(args.figures_dir)}']
    if table1 is not None:
        fwd = table1[table1['direction'].str.contains('->') & ~table1['direction'].str.startswith('PM2.5') & ~table1['direction'].str.startswith('Nightlight->Pop')]
        conclusion.append(f"Mean forward rho improvement: {table1['rho_improvement'].mean():+.4f}")
    with open(os.path.join(args.results_dir, 'conclusion.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(conclusion))
    print('\n' + '\n'.join(conclusion))
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
