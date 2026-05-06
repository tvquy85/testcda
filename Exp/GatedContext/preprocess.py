import numpy as np


def market_state_from_closes(closes, close_col=-1, window=16):
    close_px = closes[:, :, close_col]  # (N, T)
    rets = np.log(close_px[:, 1:] / close_px[:, :-1])  # (N, T‑1)

    N, Tm1 = rets.shape
    W = window
    num_win = Tm1 - W + 1
    metrics = np.zeros((num_win, 5), dtype=np.float32)

    x = np.arange(W, dtype=np.float32)  # 0 … 15 for slope calc
    var_x = x.var()

    for k in range(num_win):
        win = rets[:, k:k + W]  # (N, W)
        idx_series = win.mean(axis=0)  # equally‑weighted index, (W,)

        # 1) mean return
        mean_ret = win.mean()

        # 2) slope (market momentum)
        cov_x = ((x - x.mean()) * (idx_series - idx_series.mean())).mean()
        slope = cov_x / var_x

        # 3) realised vol
        real_vol = idx_series.std()

        # 4) dispersion
        disp = win.std(axis=0).mean()

        # 5) PCA
        cov_stk = np.cov(win, bias=True)  # (N,N), rows=stocks
        eigvals = np.linalg.eigvalsh(cov_stk)  # ascending
        pca_ratio = eigvals[-1] / eigvals.sum()

        metrics[k] = [mean_ret, slope, real_vol, disp, pca_ratio]

    min_vals = metrics.min(axis=0)
    max_vals = metrics.max(axis=0)
    metrics_norm = (metrics - min_vals) / (max_vals - min_vals)

    return metrics_norm
