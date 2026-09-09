from __future__ import annotations
import glob
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
POP_PATTERNS = ['usa_pop_{year}_CN_100m_R2025A_v1.tif', 'population_{year}.tif', 'pop_{year}.tif', 'usa_pop_{year}*.tif']
NL_PATTERNS = ['nightlight_{year}.tif', 'nl_{year}.tif', 'VNL_npp_{year}_global_vcmslcfg_v2.average_masked.tif', 'VNL_npp_{year}_global_vcmslcfg_v2.average_masked.dat.tif', 'VNL_v22_npp-j01_{year}_global_vcmslcfg.average_masked.dat.tif', 'VNL*{year}*.tif']
PM25_DEFAULT = 'UnitedStatesPM25-V5GL0502-Annual-REGIONAL-1998-2023-wThresFrac.csv'
PREPROCESS_VERSION = '2.4'
POP_YEARS = (2022, 2023, 2024)
NL_YEARS = (2022, 2023, 2024, 2025)
TRAIN_YEARS = (2022, 2023)
TEST_YEARS = (2024, 2025)
CONUS_BBOX_5070 = (-2400000, -1600000, 2300000, 1600000)

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _default_data_dir() -> str:
    return os.path.join(_project_root(), 'datasets')

def _default_cache_dir() -> str:
    return os.path.join(_project_root(), 'data_cache')

@dataclass
class PreprocessConfig:
    data_dir: str = field(default_factory=_default_data_dir)
    cache_dir: str = field(default_factory=_default_cache_dir)
    cell_size_km: float = 1.0
    crs: str = 'EPSG:5070'
    pm25_year: int = 2023
    normalize: bool = False
    seed: int = 44
    max_cells: Optional[int] = None

@dataclass
class PreprocessedUSA:
    df: pd.DataFrame
    cell_size_km: float
    years: Tuple[int, ...] = field(default_factory=tuple)
    source: str = 'real'

    @property
    def n(self) -> int:
        return len(self.df)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.df.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str) -> 'PreprocessedUSA':
        df = pd.read_parquet(path)
        meta_path = path.replace('.parquet', '_meta.json')
        cell_size_km = 1.0
        years: Tuple[int, ...] = ()
        source = 'real'
        if os.path.exists(meta_path):
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            cell_size_km = meta.get('cell_size_km', 1.0)
            years = tuple(meta.get('years', []))
            source = meta.get('source', 'real')
        return cls(df=df, cell_size_km=cell_size_km, years=years, source=source)

def _find_file(data_dir: str, patterns: Sequence[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            return hits[0]
    return None

def find_population_raster(data_dir: str, year: int) -> Optional[str]:
    for pat in POP_PATTERNS:
        path = _find_file(data_dir, [pat.format(year=year)])
        if path:
            return path
    return None

def find_nightlight_raster(data_dir: str, year: int) -> Optional[str]:
    for pat in NL_PATTERNS:
        path = _find_file(data_dir, [pat.format(year=year)])
        if path:
            return path
    return None

def _make_grid_transform(cell_size_m: float):
    (xmin, ymin, xmax, ymax) = CONUS_BBOX_5070
    width = int((xmax - xmin) / cell_size_m)
    height = int((ymax - ymin) / cell_size_m)
    transform = (xmin, cell_size_m, 0.0, ymax, 0.0, -cell_size_m)
    return ('EPSG:5070', transform, width, height, CONUS_BBOX_5070)

def _read_file_magic(path: str, n: int=16) -> bytes:
    with open(path, 'rb') as f:
        return f.read(n)

def probe_raster_file(path: str) -> dict:
    info = {'path': path, 'exists': os.path.exists(path), 'size_mb': None, 'kind': 'unknown', 'hint': ''}
    if not info['exists']:
        info['kind'] = 'missing'
        info['hint'] = 'File not found.'
        return info
    info['size_mb'] = round(os.path.getsize(path) / 1000000.0, 2)
    magic = _read_file_magic(path)
    if magic[:2] == b'\x1f\x8b':
        info['kind'] = 'gzip'
        info['hint'] = 'Gzip-compressed GeoTIFF (common for VIIRS VNL). Will auto-open via /vsigzip/ or decompress.'
    elif magic[:4] in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM+\x00'):
        info['kind'] = 'geotiff' if magic[2:3] == b'*' else 'bigtiff'
        info['hint'] = 'GeoTIFF header detected.' if info['kind'] == 'geotiff' else 'BigTIFF header detected (large rasters; GDAL/rasterio supported).'
    elif magic[:1] in (b'<', b'{'):
        info['kind'] = 'html_or_json'
        info['hint'] = 'File looks like HTML/JSON, not a raster — re-download from EOG (login required); download may have failed.'
    elif magic[:4] == b'HDF5':
        info['kind'] = 'hdf5'
        info['hint'] = 'HDF5 file — convert to GeoTIFF with gdal_translate first.'
    else:
        info['hint'] = f'Unknown header bytes: {magic[:8]!r}. Try: gdalinfo {path}'
    return info

def _gdal_translate_to_cache(src_path: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    dst = os.path.join(cache_dir, f'{base}_converted.tif')
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src_path):
        return dst
    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError('Cannot read this raster format. Install GDAL: conda install -c conda-forge gdal  OR  apt install gdal-bin libgdal-dev') from None
    magic = _read_file_magic(src_path)
    src = src_path
    if magic[:2] == b'\x1f\x8b':
        src = f'/vsigzip/{os.path.abspath(src_path)}'
    ds = gdal.Open(src)
    if ds is None:
        raise RuntimeError(f'GDAL cannot open {src_path!r}')
    opts = gdal.TranslateOptions(format='GTiff', creationOptions=['COMPRESS=NONE', 'TILED=YES'])
    out = gdal.Translate(dst, ds, options=opts)
    if out is None:
        raise RuntimeError(f'gdal_translate failed for {src_path!r}')
    out.FlushCache()
    out = None
    ds = None
    return dst

def _resolve_raster_path(src_path: str, cache_dir: str='./data_cache') -> str:
    import rasterio
    probe = probe_raster_file(src_path)
    if probe['kind'] == 'html_or_json':
        raise RuntimeError(f'{src_path!r} is not a GeoTIFF (looks like a failed web download). Re-download from https://eogdata.mines.edu/ after logging in.')
    if probe['kind'] == 'missing':
        raise FileNotFoundError(src_path)
    try:
        with rasterio.open(src_path) as ds:
            if ds.width > 0 and ds.height > 0:
                return src_path
    except Exception:
        pass
    magic = _read_file_magic(src_path)
    if magic[:2] == b'\x1f\x8b':
        vsigz = f'/vsigzip/{os.path.abspath(src_path)}'
        try:
            with rasterio.open(vsigz) as ds:
                if ds.width > 0:
                    return vsigz
        except Exception:
            pass
        return _gdal_translate_to_cache(src_path, os.path.join(cache_dir, 'raster_converted'))
    return _gdal_translate_to_cache(src_path, os.path.join(cache_dir, 'raster_converted'))

def _reproject_resample(src_path: str, dst_crs: str, dst_transform, dst_width: int, dst_height: int, cache_dir: str='./data_cache') -> np.ndarray:
    import rasterio
    from rasterio.warp import reproject, Resampling
    readable = _resolve_raster_path(src_path, cache_dir=cache_dir)
    with rasterio.open(readable) as src:
        arr = np.empty((dst_height, dst_width), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=arr, src_transform=src.transform, src_crs=src.crs, dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.average, src_nodata=src.nodata, dst_nodata=np.nan)
    return arr

def _load_pm25_by_state(csv_path: str, year: int=2023) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    pm_col = next((c for c in df.columns if 'Geographic-Mean PM2.5' in c), 'Geographic-Mean PM2.5 [ug/m3]')
    sub = df[df['Year'] == year][['Region', pm_col]].dropna()
    return dict(zip(sub['Region'], sub[pm_col].astype(float)))
_US_STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {'Alabama': (-86.9, 32.8), 'Alaska': (-152.4, 64.2), 'Arizona': (-111.7, 34.3), 'Arkansas': (-92.4, 34.8), 'California': (-119.6, 37.2), 'Colorado': (-105.5, 39.0), 'Connecticut': (-72.7, 41.6), 'Delaware': (-75.5, 39.0), 'District of Columbia': (-77.0, 38.9), 'Florida': (-81.7, 28.6), 'Georgia': (-83.4, 32.7), 'Hawaii': (-157.5, 21.3), 'Idaho': (-114.7, 44.4), 'Illinois': (-89.4, 40.0), 'Indiana': (-86.3, 39.8), 'Iowa': (-93.5, 42.0), 'Kansas': (-98.5, 38.5), 'Kentucky': (-84.9, 37.8), 'Louisiana': (-92.0, 31.0), 'Maine': (-69.4, 45.4), 'Maryland': (-76.8, 39.0), 'Massachusetts': (-71.5, 42.4), 'Michigan': (-84.5, 44.3), 'Minnesota': (-94.2, 46.3), 'Mississippi': (-89.7, 32.7), 'Missouri': (-92.5, 38.5), 'Montana': (-110.5, 47.0), 'Nebraska': (-99.9, 41.5), 'Nevada': (-116.6, 39.3), 'New Hampshire': (-71.6, 43.7), 'New Jersey': (-74.5, 40.2), 'New Mexico': (-106.0, 34.5), 'New York': (-75.5, 43.0), 'North Carolina': (-79.0, 35.5), 'North Dakota': (-100.5, 47.5), 'Ohio': (-82.8, 40.3), 'Oklahoma': (-97.5, 35.5), 'Oregon': (-120.5, 44.0), 'Pennsylvania': (-77.2, 40.9), 'Rhode Island': (-71.5, 41.7), 'South Carolina': (-80.9, 33.9), 'South Dakota': (-100.2, 44.4), 'Tennessee': (-86.6, 35.8), 'Texas': (-99.3, 31.5), 'Utah': (-111.5, 39.3), 'Vermont': (-72.7, 44.0), 'Virginia': (-78.7, 37.5), 'Washington': (-120.5, 47.4), 'West Virginia': (-80.6, 38.6), 'Wisconsin': (-89.6, 44.6), 'Wyoming': (-107.5, 43.0)}

def _assign_state(lon: float, lat: float) -> str:
    (best, best_d) = ('California', float('inf'))
    for (state, (slon, slat)) in _US_STATE_CENTROIDS.items():
        d = (lon - slon) ** 2 + (lat - slat) ** 2
        if d < best_d:
            (best_d, best) = (d, state)
    return best

def _cell_key(x_5070: np.ndarray, y_5070: np.ndarray, cell_m: float) -> np.ndarray:
    bx = np.round(x_5070 / cell_m).astype(np.int64)
    by = np.round(y_5070 / cell_m).astype(np.int64)
    return bx * 10000000 + by

def _projected_to_lonlat(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    try:
        from pyproj import Transformer
        t = Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True)
        (lon, lat) = t.transform(xs, ys)
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
    except ImportError:
        (cx, cy) = (xs.mean(), ys.mean())
        lon = -100 + (xs - cx) / 2000000 * 55
        lat = 39 + (ys - cy) / 1200000 * 12
    bad = ~np.isfinite(lon) | ~np.isfinite(lat)
    if bad.any():
        lon = np.where(bad, np.nan, lon)
        lat = np.where(bad, np.nan, lat)
    return (lon, lat)

def _synthetic_year_grid(cell_size_km: float=1.0, n_target: int=8000, seed: int=44, year_offset: int=0) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed + year_offset)
    cell_m = cell_size_km * 1000.0
    (_, transform, width, height, bbox) = _make_grid_transform(cell_m)
    (xmin, ymin, xmax, ymax) = bbox
    (cols, rows) = np.meshgrid(np.arange(width), np.arange(height))
    xs = xmin + (cols.ravel() + 0.5) * cell_m
    ys = ymax - (rows.ravel() + 0.5) * cell_m
    coords = np.column_stack([xs, ys])
    (cx, cy) = (coords[:, 0].mean(), coords[:, 1].mean())
    mask = ((coords[:, 0] - cx) / 2000000) ** 2 + ((coords[:, 1] - cy) / 1200000) ** 2 < 1.0
    coords = coords[mask]
    if n_target and coords.shape[0] > n_target:
        idx = np.random.default_rng(seed).choice(coords.shape[0], size=n_target, replace=False)
        coords = coords[idx]
    n = coords.shape[0]
    k = min(8, n - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    (_, nidx) = nbr.kneighbors(coords)
    nidx = nidx[:, 1:]
    W = np.zeros((n, n))
    for i in range(n):
        W[i, nidx[i]] = 1.0 / k
    n_clusters = max(20, n // 200)
    centers = rng.uniform(coords.min(0), coords.max(0), (n_clusters, 2))
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    pop = np.exp(-d2.min(1) / 500000 ** 2) * 5000 + rng.exponential(50, n)
    pop = np.linalg.solve(np.eye(n) - 0.3 * W, pop)
    pop *= 1.0 + 0.02 * year_offset
    nl_signal = 0.6 * pop + 0.25 * (W @ pop)
    nightlight = nl_signal + 0.15 * rng.normal(0, pop.std(), n)
    pm_signal = 0.4 * nightlight + 0.2 * pop + 0.15 * (W @ nightlight)
    pm25 = pm_signal + 0.2 * rng.normal(0, pm_signal.std(), n)
    (lon, lat) = _projected_to_lonlat(coords[:, 0], coords[:, 1])
    return pd.DataFrame({'cell_id': np.arange(n, dtype=int), 'x_5070': coords[:, 0], 'y_5070': coords[:, 1], 'longitude': lon, 'latitude': lat, 'population': pop, 'nightlight': nightlight, 'pm25': pm25})

def _dedupe_by_cell_key(df: pd.DataFrame, cell_m: float, value_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out['key'] = _cell_key(out['x_5070'].values, out['y_5070'].values, cell_m)
    agg = {'x_5070': 'mean', 'y_5070': 'mean'}
    for c in value_cols:
        agg[c] = 'mean'
    if 'longitude' in out.columns:
        agg['longitude'] = 'mean'
    if 'latitude' in out.columns:
        agg['latitude'] = 'mean'
    if 'pm25' in out.columns:
        agg['pm25'] = 'mean'
    return out.groupby('key', as_index=False).agg(agg)

def _reproject_single_band(src_path: str, cell_size_km: float, cache_dir: str) -> Tuple[np.ndarray, object, str]:
    from rasterio.transform import from_origin
    cell_m = cell_size_km * 1000.0
    (crs, _, width, height, bbox) = _make_grid_transform(cell_m)
    (xmin, ymin, xmax, ymax) = bbox
    transform = from_origin(xmin, ymax, cell_m, cell_m)
    arr = _reproject_resample(src_path, crs, transform, width, height, cache_dir=cache_dir)
    return (arr, transform, crs)

def _load_pop_layer(cfg: PreprocessConfig, year: int) -> Tuple[pd.DataFrame, str]:
    from rasterio.transform import xy as rio_xy
    pop_path = find_population_raster(cfg.data_dir, year)
    if pop_path is None:
        raise FileNotFoundError(f'population raster for {year} not found')
    cell_m = cfg.cell_size_km * 1000.0
    (pop_arr, transform, _) = _reproject_single_band(pop_path, cfg.cell_size_km, cfg.cache_dir)
    (rows, cols) = np.where(np.isfinite(pop_arr) & (pop_arr > 0))
    (xs, ys) = rio_xy(transform, rows, cols, offset='center')
    df = pd.DataFrame({'x_5070': np.asarray(xs, float), 'y_5070': np.asarray(ys, float), 'population': pop_arr[rows, cols]})
    df = _dedupe_by_cell_key(df, cell_m, ['population'])
    return (df, 'real')

def _load_nl_layer(cfg: PreprocessConfig, year: int) -> Tuple[pd.DataFrame, str]:
    from rasterio.transform import xy as rio_xy
    nl_path = find_nightlight_raster(cfg.data_dir, year)
    if nl_path is None:
        raise FileNotFoundError(f'nightlight raster for {year} not found')
    cell_m = cfg.cell_size_km * 1000.0
    (nl_arr, transform, _) = _reproject_single_band(nl_path, cfg.cell_size_km, cfg.cache_dir)
    (rows, cols) = np.where(np.isfinite(nl_arr) & (nl_arr > 0))
    (xs, ys) = rio_xy(transform, rows, cols, offset='center')
    df = pd.DataFrame({'x_5070': np.asarray(xs, float), 'y_5070': np.asarray(ys, float), 'nightlight': nl_arr[rows, cols]})
    df = _dedupe_by_cell_key(df, cell_m, ['nightlight'])
    return (df, 'real')

def _attach_pm25(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    pm25_path = os.path.join(cfg.data_dir, PM25_DEFAULT)
    pm25_lookup = _load_pm25_by_state(pm25_path, year=cfg.pm25_year) if os.path.exists(pm25_path) else {}
    pm_default = float(np.nanmean(list(pm25_lookup.values()))) if pm25_lookup else 10.0
    (lon, lat) = _projected_to_lonlat(df['x_5070'].values, df['y_5070'].values)
    out = df.copy()
    out['longitude'] = lon
    out['latitude'] = lat
    out['pm25'] = [pm25_lookup.get(_assign_state(lo, la), pm_default) for (lo, la) in zip(lon, lat)]
    valid = np.isfinite(out['longitude']) & np.isfinite(out['latitude']) & np.isfinite(out['pm25']) & (out['longitude'] >= -125) & (out['longitude'] <= -66) & (out['latitude'] >= 24) & (out['latitude'] <= 50)
    return out.loc[valid].reset_index(drop=True)

def _maybe_subsample(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    if cfg.max_cells and len(df) > cfg.max_cells:
        n_before = len(df)
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(n_before, size=cfg.max_cells, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        warnings.warn(f'Subsampled grid from {n_before:,} to {cfg.max_cells:,} cells (max_cells).')
    return df

def _raster_to_dataframe(pop_path: str, nl_path: str, pm25_lookup: Dict[str, float], cell_size_km: float, cache_dir: str='./data_cache') -> pd.DataFrame:
    import rasterio
    from rasterio.transform import from_origin, xy as rio_xy
    cell_m = cell_size_km * 1000.0
    (crs, _, width, height, bbox) = _make_grid_transform(cell_m)
    (xmin, ymin, xmax, ymax) = bbox
    transform = from_origin(xmin, ymax, cell_m, cell_m)
    pop_arr = _reproject_resample(pop_path, crs, transform, width, height, cache_dir=cache_dir)
    nl_arr = _reproject_resample(nl_path, crs, transform, width, height, cache_dir=cache_dir)
    (rows, cols) = np.where(np.isfinite(pop_arr) & np.isfinite(nl_arr) & (pop_arr > 0))
    pop_vals = pop_arr[rows, cols]
    nl_vals = nl_arr[rows, cols]
    (xs, ys) = rio_xy(transform, rows, cols, offset='center')
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    (lon, lat) = _projected_to_lonlat(xs, ys)
    pm_default = float(np.nanmean(list(pm25_lookup.values()))) if pm25_lookup else 10.0
    pm25_vals = np.array([pm25_lookup.get(_assign_state(lo, la), pm_default) for (lo, la) in zip(lon, lat)])
    valid = np.isfinite(pop_vals) & np.isfinite(nl_vals) & np.isfinite(pm25_vals) & np.isfinite(lon) & np.isfinite(lat) & (lon >= -125) & (lon <= -66) & (lat >= 24) & (lat <= 50)
    df = pd.DataFrame({'cell_id': np.arange(valid.sum(), dtype=int), 'x_5070': xs[valid], 'y_5070': ys[valid], 'longitude': lon[valid], 'latitude': lat[valid], 'population': pop_vals[valid], 'nightlight': nl_vals[valid], 'pm25': pm25_vals[valid]})
    cell_m = cell_size_km * 1000.0
    df = _dedupe_by_cell_key(df, cell_m, ['population', 'nightlight', 'pm25'])
    df = df.drop(columns=['key'])
    df['cell_id'] = np.arange(len(df), dtype=int)
    return df

def _normalize_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        v = out[c].values.astype(float)
        s = v.std()
        out[c] = (v - v.mean()) / s if s > 1e-12 else v - v.mean()
    return out

def preprocess_single_year(cfg: PreprocessConfig, year: int, nl_year: Optional[int]=None) -> PreprocessedUSA:
    nl_year = nl_year or year
    pop_path = find_population_raster(cfg.data_dir, year)
    nl_path = find_nightlight_raster(cfg.data_dir, nl_year)
    pm25_path = os.path.join(cfg.data_dir, PM25_DEFAULT)
    if pop_path is None or nl_path is None:
        warnings.warn(f'Rasters for year {year} not found in {cfg.data_dir}; using synthetic surrogate.')
        df = _synthetic_year_grid(cell_size_km=cfg.cell_size_km, seed=cfg.seed, year_offset=year - 2022)
        source = 'synthetic'
    else:
        try:
            pm25_lookup = _load_pm25_by_state(pm25_path, year=cfg.pm25_year) if os.path.exists(pm25_path) else {}
            df = _raster_to_dataframe(pop_path, nl_path, pm25_lookup, cfg.cell_size_km, cache_dir=cfg.cache_dir)
            source = 'real'
        except ImportError:
            warnings.warn('rasterio not installed; using synthetic USA grid.')
            df = _synthetic_year_grid(cell_size_km=cfg.cell_size_km, seed=cfg.seed, year_offset=year - 2022)
            source = 'synthetic'
        except Exception as exc:
            warnings.warn(f'Raster load failed for pop={pop_path!r}, nl={nl_path!r}: {exc}. Using synthetic grid for this year.')
            df = _synthetic_year_grid(cell_size_km=cfg.cell_size_km, seed=cfg.seed, year_offset=year - 2022)
            source = 'synthetic'
    df = df.dropna(subset=['population', 'nightlight', 'pm25', 'longitude', 'latitude'])
    if cfg.normalize:
        df = _normalize_df(df, ['population', 'nightlight', 'pm25'])
    df = _maybe_subsample(df, cfg)
    df['cell_id'] = np.arange(len(df), dtype=int)
    return PreprocessedUSA(df=df, cell_size_km=cfg.cell_size_km, years=(year,), source=source)

def preprocess_multi_year_average(cfg: PreprocessConfig, pop_years: Sequence[int], nl_years: Sequence[int]) -> PreprocessedUSA:
    if not pop_years or not nl_years:
        raise ValueError('pop_years and nl_years must be non-empty')
    cell_m = cfg.cell_size_km * 1000.0
    sources: List[str] = []
    pop_parts = []
    for y in pop_years:
        try:
            (d, src) = _load_pop_layer(cfg, y)
            sources.append(src)
            pop_parts.append(d[['key', 'population']].rename(columns={'population': f'pop_{y}'}))
        except Exception as exc:
            warnings.warn(f'population {y} load failed ({exc}); skipping.')
    if not pop_parts:
        raise RuntimeError('No population years could be loaded.')
    pop_wide = pop_parts[0]
    for part in pop_parts[1:]:
        pop_wide = pop_wide.merge(part, on='key', how='inner')
    pop_cols = [c for c in pop_wide.columns if c.startswith('pop_')]
    pop_wide['population'] = pop_wide[pop_cols].mean(axis=1)
    nl_parts = []
    for y in nl_years:
        try:
            (d, src) = _load_nl_layer(cfg, y)
            sources.append(src)
            nl_parts.append(d[['key', 'nightlight']].rename(columns={'nightlight': f'nl_{y}'}))
        except Exception as exc:
            warnings.warn(f'nightlight {y} load failed ({exc}); skipping.')
    if not nl_parts:
        raise RuntimeError('No nightlight years could be loaded.')
    nl_wide = nl_parts[0]
    for part in nl_parts[1:]:
        nl_wide = nl_wide.merge(part, on='key', how='inner')
    nl_cols = [c for c in nl_wide.columns if c.startswith('nl_')]
    nl_wide['nightlight'] = nl_wide[nl_cols].mean(axis=1)
    (coord_ref, _) = _load_pop_layer(cfg, pop_years[0])
    merged = pop_wide[['key', 'population']].merge(nl_wide[['key', 'nightlight']], on='key', how='inner').merge(coord_ref[['key', 'x_5070', 'y_5070']], on='key', how='inner')
    merged = _attach_pm25(merged, cfg)
    merged = merged.drop(columns=['key'])
    merged['cell_id'] = np.arange(len(merged), dtype=int)
    merged = merged[['cell_id', 'x_5070', 'y_5070', 'longitude', 'latitude', 'population', 'nightlight', 'pm25']]
    if cfg.normalize:
        merged = _normalize_df(merged, ['population', 'nightlight', 'pm25'])
    merged = _maybe_subsample(merged, cfg)
    merged['cell_id'] = np.arange(len(merged), dtype=int)
    source = 'real' if sources and all((s == 'real' for s in sources)) else 'synthetic'
    return PreprocessedUSA(df=merged, cell_size_km=cfg.cell_size_km, years=tuple(sorted(set(pop_years) | set(nl_years))), source=source)

def _filter_available_years(cfg: PreprocessConfig, years: Sequence[int], kind: str) -> List[int]:
    finder = find_population_raster if kind == 'pop' else find_nightlight_raster
    available = []
    for y in years:
        if finder(cfg.data_dir, y):
            available.append(y)
        else:
            warnings.warn(f'{kind} raster for {y} not found in {cfg.data_dir}; skipping that year in the temporal average.')
    return available

def run_preprocessing(cfg: Optional[PreprocessConfig]=None, save: bool=True) -> Dict[str, PreprocessedUSA]:
    cfg = cfg or PreprocessConfig()
    os.makedirs(cfg.cache_dir, exist_ok=True)
    train_pop = _filter_available_years(cfg, [2022, 2023], 'pop') or [2022, 2023]
    train_nl = _filter_available_years(cfg, [2022, 2023], 'nl') or [2022, 2023]
    test_pop = _filter_available_years(cfg, [2024], 'pop') or [2024]
    test_nl = _filter_available_years(cfg, [2024, 2025], 'nl') or [2024]
    outputs = {'train': preprocess_multi_year_average(cfg, pop_years=train_pop, nl_years=train_nl), 'test': preprocess_multi_year_average(cfg, pop_years=test_pop, nl_years=test_nl), '2023': preprocess_single_year(cfg, 2023)}
    if save:
        import json
        for (name, grid) in outputs.items():
            path = os.path.join(cfg.cache_dir, f'usa_{name}_{cfg.cell_size_km:.0f}km.parquet')
            grid.save(path)
            meta = {'cell_size_km': grid.cell_size_km, 'years': list(grid.years), 'source': grid.source, 'n': grid.n}
            with open(path.replace('.parquet', '_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
    return outputs

def load_preprocessed(cache_dir: Optional[str]=None, split: str='2023', cell_size_km: float=1.0) -> PreprocessedUSA:
    cache_dir = cache_dir or _default_cache_dir()
    path = os.path.join(cache_dir, f'usa_{split}_{cell_size_km:.0f}km.parquet')
    if os.path.exists(path):
        return PreprocessedUSA.load(path)
    cfg = PreprocessConfig(cache_dir=cache_dir, cell_size_km=cell_size_km)
    if split == 'train':
        return preprocess_multi_year_average(cfg, pop_years=[2022, 2023], nl_years=[2022, 2023])
    if split == 'test':
        return preprocess_multi_year_average(cfg, pop_years=[2024], nl_years=[2024, 2025])
    return preprocess_single_year(cfg, 2023)

def diagnose_data_setup(cfg: Optional[PreprocessConfig]=None) -> bool:
    cfg = cfg or PreprocessConfig()
    ok = True
    print('=' * 60)
    print(f'USA preprocessing diagnostics  (code version {PREPROCESS_VERSION})')
    print('=' * 60)
    this_file = os.path.abspath(__file__)
    with open(this_file, 'r', encoding='utf-8') as f:
        src = f.read()
    if 'xmin, ymin, xmax, ymax = bbox' in src and '_project_root' in src:
        print(f'[ OK ] preprocess_usa.py is version {PREPROCESS_VERSION}')
    else:
        print('[FAIL] Outdated preprocess_usa.py — sync latest src/preprocess_usa.py')
        ok = False
    try:
        import rasterio
        print('[ OK ] rasterio is installed')
    except ImportError:
        print('[FAIL] rasterio NOT installed — real GeoTIFFs cannot be read')
        print('       Fix: pip install rasterio')
        ok = False
    try:
        import pyproj
        print('[ OK ] pyproj is installed')
    except ImportError:
        print('[WARN] pyproj not installed — lon/lat may be approximate')
    print(f'\nData directory: {os.path.abspath(cfg.data_dir)}')
    if not os.path.isdir(cfg.data_dir):
        print(f'[FAIL] Directory does not exist: {cfg.data_dir}')
        ok = False
    else:
        tifs = sorted(glob.glob(os.path.join(cfg.data_dir, '*.tif')))
        print(f'       Found {len(tifs)} .tif file(s)')
        for t in tifs[:8]:
            print(f'         - {os.path.basename(t)}')
        if len(tifs) > 8:
            print(f'         ... and {len(tifs) - 8} more')
    pm25_path = os.path.join(cfg.data_dir, PM25_DEFAULT)
    if os.path.exists(pm25_path):
        print(f'[ OK ] PM2.5 CSV found')
    else:
        print(f'[WARN] PM2.5 CSV missing: {PM25_DEFAULT}')
    print('\nRaster discovery (pop / nightlight):')
    required_years = [2022, 2023, 2024]
    optional_years = [2025]
    for y in required_years + optional_years:
        pop = find_population_raster(cfg.data_dir, y)
        nl = find_nightlight_raster(cfg.data_dir, y)
        pop_s = os.path.basename(pop) if pop else 'NOT FOUND'
        nl_s = os.path.basename(nl) if nl else 'NOT FOUND'
        need_pop = y in required_years
        year_ok = (pop or not need_pop) and nl
        status = '[ OK ]' if year_ok else '[WARN]'
        print(f'  {status} {y}: pop={pop_s}  |  nl={nl_s}')
        for (label, path) in [('pop', pop), ('nl', nl)]:
            if not path:
                continue
            pi = probe_raster_file(path)
            print(f"         {label} format: {pi['kind']} ({pi['size_mb']} MB) — {pi['hint']}")
            if pi['kind'] in ('gzip', 'html_or_json'):
                ok = False
            elif pi['kind'] == 'unknown':
                ok = False
        if y in required_years and (not pop or not nl):
            ok = False
        elif y in optional_years and (not nl):
            print(f'         (2025 nightlight optional for test split)')
    try:
        from osgeo import gdal
        print('\n[ OK ] GDAL Python bindings available (needed for VIIRS VNL gzip)')
    except ImportError:
        print('\n[WARN] GDAL Python bindings not found.')
        print('       VIIRS .dat.tif files are often gzip GeoTIFFs.')
        print('       Fix: conda install -c conda-forge gdal rasterio')
        print('       Or:  apt install gdal-bin libgdal-dev && pip install gdal==$(gdal-config --version)')
    print('\n' + '=' * 60)
    if ok:
        print('All critical checks passed. Run: python preprocess_usa.py')
    else:
        print('Some checks FAILED — fix the items above before preprocessing.')
        print("If you still see 'expected 5, got 4', you are running OLD code.")
        print('Verify: grep PREPROCESS_VERSION src/preprocess_usa.py')
    print('=' * 60)
    return ok
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='USA preprocessing (run from any directory)')
    p.add_argument('--diagnose', action='store_true', help='Check data files only')
    p.add_argument('--data-dir', default=None, help='Override datasets path')
    p.add_argument('--cache-dir', default=None, help='Override cache path')
    args = p.parse_args()
    cfg = PreprocessConfig()
    if args.data_dir:
        cfg.data_dir = os.path.abspath(args.data_dir)
    if args.cache_dir:
        cfg.cache_dir = os.path.abspath(args.cache_dir)
    print(f'preprocess_usa version {PREPROCESS_VERSION}')
    print(f'Project root: {_project_root()}')
    print(f'Data dir:     {cfg.data_dir}')
    print(f'Cache dir:    {cfg.cache_dir}')
    if args.diagnose:
        raise SystemExit(0 if diagnose_data_setup(cfg) else 1)
    out = run_preprocessing(cfg)
    for (k, v) in out.items():
        print(f'{k}: n={v.n}, source={v.source}, years={v.years}')
