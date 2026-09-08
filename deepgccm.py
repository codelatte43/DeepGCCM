from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from .metrics import ConvergenceCurve, pearson_r, rmse

def build_knn_graph(coords: np.ndarray, k: int=8) -> np.ndarray:
    n = coords.shape[0]
    k = min(k, n - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    (_, idx) = nbr.kneighbors(coords)
    return idx[:, 1:]

class GATLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, heads: int=4, dropout: float=0.1, leaky_slope: float=0.2, concat: bool=True):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout
        self.leaky_slope = leaky_slope
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_self = nn.Parameter(torch.empty(1, 1, heads, out_dim))
        self.a_nbr = nn.Parameter(torch.empty(1, 1, heads, out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_self)
        nn.init.xavier_uniform_(self.a_nbr)

    def forward(self, x: torch.Tensor, nbr_idx: torch.Tensor, return_alpha: bool=False):
        (N, K) = nbr_idx.shape
        (H, D) = (self.heads, self.out_dim)
        h = self.W(x).view(N, H, D)
        h_nbr = h[nbr_idx]
        h_self = h.unsqueeze(1).expand(N, K, H, D)
        e_self = (h_self * self.a_self).sum(-1)
        e_nbr = (h_nbr * self.a_nbr).sum(-1)
        e = F.leaky_relu(e_self + e_nbr, negative_slope=self.leaky_slope)
        alpha = F.softmax(e, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = (alpha.unsqueeze(-1) * h_nbr).sum(dim=1)
        if self.concat:
            emb = out.reshape(N, H * D)
        else:
            emb = out.mean(dim=1)
        if return_alpha:
            return (emb, alpha.mean(dim=2))
        return emb

class SpatialEncoder(nn.Module):

    def __init__(self, embed_dim: int=32, hidden: int=32, heads: int=4, n_layers: int=2, dropout: float=0.1, use_gnn: bool=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.input_proj = nn.Linear(1, hidden)
        if use_gnn:
            assert hidden % heads == 0, 'hidden must be divisible by heads'
            head_dim = hidden // heads
            self.layers = nn.ModuleList([GATLayer(hidden, head_dim, heads=heads, dropout=dropout, concat=True) for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.output = nn.Linear(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, nbr_idx: torch.Tensor, return_alpha: bool=False):
        h = F.elu(self.input_proj(x))
        alpha_out = None
        for layer in self.layers:
            if self.use_gnn:
                if return_alpha and isinstance(layer, GATLayer):
                    (h_new, alpha) = layer(h, nbr_idx, return_alpha=True)
                    alpha_out = alpha
                else:
                    h_new = layer(h, nbr_idx)
            else:
                h_new = layer(h)
            h = F.elu(h_new) + h
            h = self.dropout(h)
        emb = self.output(h)
        if return_alpha:
            return (emb, alpha_out)
        return emb

class CrossMapHead(nn.Module):

    def __init__(self, embed_dim: int, attn_dim: int=32, n_simplex_neighbours: int=8, use_attention: bool=True, attn_topk: Optional[int]=None):
        super().__init__()
        self.use_attention = use_attention
        self.k_simplex = n_simplex_neighbours
        self.attn_topk = attn_topk
        if use_attention:
            self.q_proj = nn.Linear(embed_dim, attn_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, attn_dim, bias=False)
            self.scale = math.sqrt(attn_dim)

    @staticmethod
    def _apply_geo_ban(scores_or_dist: torch.Tensor, lib_idx: torch.Tensor, geo_ban: Optional[torch.Tensor], start: int, end: int, fill_value: float) -> torch.Tensor:
        if geo_ban is None:
            return scores_or_dist
        ban = geo_ban[start:end]
        banned = (lib_idx.unsqueeze(0).unsqueeze(2) == ban.unsqueeze(1)).any(dim=2)
        return scores_or_dist.masked_fill(banned, fill_value)

    @staticmethod
    def _apply_group_ban(scores_or_dist: torch.Tensor, lib_idx: torch.Tensor, group_id: Optional[torch.Tensor], start: int, end: int, fill_value: float) -> torch.Tensor:
        if group_id is None:
            return scores_or_dist
        qg = group_id[start:end].unsqueeze(1)
        lg = group_id[lib_idx].unsqueeze(0)
        return scores_or_dist.masked_fill(qg == lg, fill_value)

    def forward(self, manifold: torch.Tensor, target: torch.Tensor, lib_idx: torch.Tensor, mask_self: bool=True, geo_ban: Optional[torch.Tensor]=None, group_id: Optional[torch.Tensor]=None) -> torch.Tensor:
        N = manifold.shape[0]
        L = lib_idx.shape[0]
        lib_emb = manifold[lib_idx]
        lib_target = target[lib_idx]
        if self.use_attention:
            q = self.q_proj(manifold)
            k = self.k_proj(lib_emb)
            chunk = 2048 if N * L > 20000000 else N
            preds = []
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                scores = q[start:end] @ k.t() / self.scale
                if mask_self:
                    self_mask = torch.arange(start, end, device=manifold.device).unsqueeze(1) == lib_idx.unsqueeze(0)
                    scores = scores.masked_fill(self_mask, -1000000000.0)
                scores = self._apply_geo_ban(scores, lib_idx, geo_ban, start, end, fill_value=-1000000000.0)
                scores = self._apply_group_ban(scores, lib_idx, group_id, start, end, fill_value=-1000000000.0)
                if self.attn_topk is not None and self.attn_topk < L:
                    kk = min(self.attn_topk, L)
                    (top_scores, top_idx) = torch.topk(scores, k=kk, dim=1)
                    alpha = F.softmax(top_scores, dim=1)
                    preds.append((alpha * lib_target[top_idx]).sum(dim=1))
                else:
                    alpha = F.softmax(scores, dim=1)
                    preds.append(alpha @ lib_target)
            return torch.cat(preds, dim=0)
        else:
            d2 = ((manifold ** 2).sum(1, keepdim=True) + (lib_emb ** 2).sum(1) - 2 * manifold @ lib_emb.t()).clamp_min(1e-12)
            dist = torch.sqrt(d2)
            if mask_self:
                self_mask = torch.arange(N, device=manifold.device).unsqueeze(1) == lib_idx.unsqueeze(0)
                dist = dist.masked_fill(self_mask, float('inf'))
            dist = self._apply_geo_ban(dist, lib_idx, geo_ban, 0, N, fill_value=float('inf'))
            dist = self._apply_group_ban(dist, lib_idx, group_id, 0, N, fill_value=float('inf'))
            banned = 0 if geo_ban is None else geo_ban.shape[1]
            k = min(self.k_simplex, max(1, L - (1 if mask_self else 0) - banned))
            (d_top, idx_top) = torch.topk(dist, k=k, dim=1, largest=False)
            d_min = d_top[:, :1] + 1e-12
            w = torch.exp(-d_top / d_min)
            w = w / w.sum(dim=1, keepdim=True)
            return (w * lib_target[idx_top]).sum(dim=1)

class DeepGCCM(nn.Module):

    def __init__(self, embed_dim: int=32, hidden: int=32, attn_dim: int=32, heads: int=4, n_layers: int=2, dropout: float=0.1, n_simplex_neighbours: int=8, use_gnn: bool=True, use_attention: bool=True, attn_topk: Optional[int]=None):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.enc_x = SpatialEncoder(embed_dim=embed_dim, hidden=hidden, heads=heads, n_layers=n_layers, dropout=dropout, use_gnn=use_gnn)
        self.enc_y = SpatialEncoder(embed_dim=embed_dim, hidden=hidden, heads=heads, n_layers=n_layers, dropout=dropout, use_gnn=use_gnn)
        self.head_xmap_y = CrossMapHead(embed_dim=embed_dim, attn_dim=attn_dim, n_simplex_neighbours=n_simplex_neighbours, use_attention=use_attention, attn_topk=attn_topk)
        self.head_ymap_x = CrossMapHead(embed_dim=embed_dim, attn_dim=attn_dim, n_simplex_neighbours=n_simplex_neighbours, use_attention=use_attention, attn_topk=attn_topk)

    def encode(self, x: torch.Tensor, y: torch.Tensor, nbr_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.enc_x(x, nbr_idx), self.enc_y(y, nbr_idx))

    def forward(self, x: torch.Tensor, y: torch.Tensor, nbr_idx: torch.Tensor, lib_idx: torch.Tensor, geo_ban: Optional[torch.Tensor]=None, group_id: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        (Mx, My) = self.encode(x, y, nbr_idx)
        y_hat = self.head_xmap_y(Mx, y.squeeze(-1), lib_idx, geo_ban=geo_ban, group_id=group_id)
        x_hat = self.head_ymap_x(My, x.squeeze(-1), lib_idx, geo_ban=geo_ban, group_id=group_id)
        return (y_hat, x_hat)

@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 0.005
    weight_decay: float = 1e-05
    train_lib_frac: float = 0.8
    k_neighbors: int = 8
    embed_dim: int = 32
    hidden: int = 32
    heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    use_gnn: bool = True
    use_attention: bool = True
    attn_topk: Optional[int] = None
    geo_exclude_k: int = 0
    seed: int = 0
    device: Optional[str] = None
    verbose: bool = False

def _to_tensor(arr, device):
    return torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=device)

def _make_geo_ban(coords: np.ndarray, k: int, device: torch.device) -> Optional[torch.Tensor]:
    if k <= 0:
        return None
    nbr = build_knn_graph(coords, k=k)
    return torch.as_tensor(nbr, dtype=torch.long, device=device)

def train_deepgccm(x: np.ndarray, y: np.ndarray, coords: np.ndarray, config: Optional[TrainConfig]=None, group_id: Optional[np.ndarray]=None) -> Tuple[DeepGCCM, dict]:
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    n = x.shape[0]
    nbr_np = build_knn_graph(coords, k=cfg.k_neighbors)
    x_t = _to_tensor(x, device).unsqueeze(-1)
    y_t = _to_tensor(y, device).unsqueeze(-1)
    nbr_t = torch.as_tensor(nbr_np, dtype=torch.long, device=device)
    x_full = x_t.squeeze(-1)
    y_full = y_t.squeeze(-1)
    geo_ban = _make_geo_ban(coords, cfg.geo_exclude_k, device)
    group_t = torch.as_tensor(group_id, dtype=torch.long, device=device) if group_id is not None else None
    model = DeepGCCM(embed_dim=cfg.embed_dim, hidden=cfg.hidden, heads=cfg.heads, n_layers=cfg.n_layers, dropout=cfg.dropout, use_gnn=cfg.use_gnn, use_attention=cfg.use_attention, attn_topk=cfg.attn_topk).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    L_train = max(int(cfg.train_lib_frac * n), max(cfg.k_neighbors + 2, 8))
    L_train = min(L_train, n)
    log = {'loss': [], 'rho_xmap_y': [], 'rho_ymap_x': []}
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        if L_train >= n:
            lib = torch.arange(n, device=device)
        else:
            lib_np = rng.choice(n, size=L_train, replace=False)
            lib = torch.as_tensor(lib_np, dtype=torch.long, device=device)
        (y_hat, x_hat) = model(x_t, y_t, nbr_t, lib, geo_ban=geo_ban, group_id=group_t)
        loss = F.mse_loss(y_hat, y_full) + F.mse_loss(x_hat, x_full)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        log['loss'].append(loss.item())
        if cfg.verbose and (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            with torch.no_grad():
                model.eval()
                lib_full = torch.arange(n, device=device)
                (yh, xh) = model(x_t, y_t, nbr_t, lib_full, geo_ban=geo_ban, group_id=group_t)
                rxy = pearson_r(yh.cpu().numpy(), y)
                ryx = pearson_r(xh.cpu().numpy(), x)
                log['rho_xmap_y'].append(rxy)
                log['rho_ymap_x'].append(ryx)
                print(f'  [DeepGCCM] epoch {epoch + 1:4d}/{cfg.epochs}  loss={loss.item():.4f}  rho(X xmap Y)={rxy:+.3f}  rho(Y xmap X)={ryx:+.3f}')
    log['elapsed_sec'] = time.time() - t0
    log['geo_exclude_k'] = cfg.geo_exclude_k
    log['attn_topk'] = cfg.attn_topk
    return (model, log)

@torch.no_grad()
def deepgccm_convergence(model: DeepGCCM, x: np.ndarray, y: np.ndarray, coords: np.ndarray, k_neighbors: int=8, lib_sizes: Optional[Sequence[int]]=None, n_repeat: int=20, seed: int=0, verbose: bool=False, train_elapsed_sec: float=0.0, geo_exclude_k: int=0, group_id: Optional[np.ndarray]=None) -> dict:
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    n = x.shape[0]
    if lib_sizes is None:
        lo = max(2 * (k_neighbors + 1), 10)
        if lo >= n:
            lo = max(2, n // 4)
        lib_sizes = np.unique(np.linspace(lo, n, num=8, dtype=int))
    lib_sizes = np.asarray(lib_sizes, dtype=int)
    nbr_np = build_knn_graph(coords, k=k_neighbors)
    x_t = _to_tensor(x, device).unsqueeze(-1)
    y_t = _to_tensor(y, device).unsqueeze(-1)
    nbr_t = torch.as_tensor(nbr_np, dtype=torch.long, device=device)
    geo_ban = _make_geo_ban(coords, geo_exclude_k, device)
    group_t = torch.as_tensor(group_id, dtype=torch.long, device=device) if group_id is not None else None
    model.eval()
    (Mx, My) = model.encode(x_t, y_t, nbr_t)
    x_vec = x_t.squeeze(-1)
    y_vec = y_t.squeeze(-1)
    (rho_xy_mean, rho_xy_std) = ([], [])
    (rho_yx_mean, rho_yx_std) = ([], [])
    for L in lib_sizes:
        (rxy_runs, ryx_runs) = ([], [])
        reps = 1 if L >= n else n_repeat
        for _ in range(reps):
            if L >= n:
                lib = torch.arange(n, device=device)
            else:
                lib_np = rng.choice(n, size=L, replace=False)
                lib = torch.as_tensor(lib_np, dtype=torch.long, device=device)
            y_hat = model.head_xmap_y(Mx, y_vec, lib, geo_ban=geo_ban, group_id=group_t).cpu().numpy()
            x_hat = model.head_ymap_x(My, x_vec, lib, geo_ban=geo_ban, group_id=group_t).cpu().numpy()
            rxy_runs.append(pearson_r(y_hat, y))
            ryx_runs.append(pearson_r(x_hat, x))
        rho_xy_mean.append(np.mean(rxy_runs))
        rho_xy_std.append(np.std(rxy_runs))
        rho_yx_mean.append(np.mean(ryx_runs))
        rho_yx_std.append(np.std(ryx_runs))
        if verbose:
            print(f'  [DeepGCCM eval] L={L:5d}  rho(X xmap Y)={rho_xy_mean[-1]:+.3f}  rho(Y xmap X)={rho_yx_mean[-1]:+.3f}')
    lib_full = torch.arange(n, device=device)
    y_hat_full = model.head_xmap_y(Mx, y_vec, lib_full, geo_ban=geo_ban, group_id=group_t).cpu().numpy()
    x_hat_full = model.head_ymap_x(My, x_vec, lib_full, geo_ban=geo_ban, group_id=group_t).cpu().numpy()
    return {'lib_sizes': lib_sizes, 'rho_xmap_y_curve': ConvergenceCurve(lib_sizes=lib_sizes, rho_mean=np.array(rho_xy_mean), rho_std=np.array(rho_xy_std)), 'rho_ymap_x_curve': ConvergenceCurve(lib_sizes=lib_sizes, rho_mean=np.array(rho_yx_mean), rho_std=np.array(rho_yx_std)), 'rho_xmap_y_full': pearson_r(y_hat_full, y), 'rho_ymap_x_full': pearson_r(x_hat_full, x), 'rmse_xmap_y_full': rmse(y_hat_full, y), 'rmse_ymap_x_full': rmse(x_hat_full, x), 'y_hat_full': y_hat_full, 'x_hat_full': x_hat_full, 'elapsed_sec': train_elapsed_sec}

@torch.no_grad()
def extract_gat_attention(model: DeepGCCM, x: np.ndarray, y: np.ndarray, coords: np.ndarray, k_neighbors: int=8, variable: str='x') -> dict:
    device = next(model.parameters()).device
    nbr_np = build_knn_graph(coords, k=k_neighbors)
    nbr_t = torch.as_tensor(nbr_np, dtype=torch.long, device=device)
    if variable.lower() in ('x', 'population', 'pop'):
        enc = model.enc_x
        vals = _to_tensor(x, device).unsqueeze(-1)
    else:
        enc = model.enc_y
        vals = _to_tensor(y, device).unsqueeze(-1)
    enc.eval()
    (_, alpha) = enc(vals, nbr_t, return_alpha=True)
    return {'alpha': alpha.cpu().numpy(), 'nbr_idx': nbr_np, 'coords': coords, 'variable': variable}

@torch.no_grad()
def extract_cross_attention(model: DeepGCCM, x: np.ndarray, y: np.ndarray, coords: np.ndarray, k_neighbors: int=8, direction: str='xmap_y', top_k: int=50) -> dict:
    device = next(model.parameters()).device
    n = x.shape[0]
    nbr_np = build_knn_graph(coords, k=k_neighbors)
    x_t = _to_tensor(x, device).unsqueeze(-1)
    y_t = _to_tensor(y, device).unsqueeze(-1)
    nbr_t = torch.as_tensor(nbr_np, dtype=torch.long, device=device)
    model.eval()
    (Mx, My) = model.encode(x_t, y_t, nbr_t)
    lib = torch.arange(n, device=device)
    if direction == 'xmap_y':
        head = model.head_xmap_y
        manifold = Mx
        target = y_t.squeeze(-1)
    else:
        head = model.head_ymap_x
        manifold = My
        target = x_t.squeeze(-1)
    if not head.use_attention:
        return {'direction': direction, 'scores': None, 'note': 'attention disabled'}
    q = head.q_proj(manifold)
    k = head.k_proj(manifold[lib])
    scores = q @ k.t() / head.scale
    self_mask = torch.arange(n, device=device).unsqueeze(1) == lib.unsqueeze(0)
    scores = scores.masked_fill(self_mask, -1000000000.0)
    if head.attn_topk is not None and head.attn_topk < n:
        kk = min(head.attn_topk, n)
        (top_scores, top_pos) = torch.topk(scores, k=kk, dim=1)
        alpha = torch.zeros_like(scores)
        alpha.scatter_(1, top_pos, F.softmax(top_scores, dim=1))
    else:
        alpha = F.softmax(scores, dim=1)
    (top_vals, top_idx) = torch.topk(alpha, k=min(top_k, n), dim=1)
    return {'direction': direction, 'top_idx': top_idx.cpu().numpy(), 'top_alpha': top_vals.cpu().numpy(), 'mean_alpha': alpha.mean(dim=0).cpu().numpy()}
