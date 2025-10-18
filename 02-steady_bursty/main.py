#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
import yaml

# ----------------- helpers -----------------
def ema_last(x, alpha):
    return float(pd.Series(x).ewm(alpha=alpha, adjust=False).mean().iloc[-1])

def cv(x):
    x = np.asarray(x, dtype=float); m = x.mean()
    return float(x.std(ddof=0) / m) if m > 0 else float("nan")

def mad(x):
    x = np.asarray(x, dtype=float); med = np.median(x)
    return float(np.median(np.abs(x - med)))

def pctl(x, q): return float(np.quantile(np.asarray(x, dtype=float), q))

def lin_slope(x):
    y = np.asarray(x, dtype=float); t = np.arange(len(y), dtype=float)
    tc = t - t.mean(); denom = float((tc**2).sum())
    if denom == 0.0: return 0.0
    return float((tc * (y - y.mean())).sum() / denom)

def trend_flag(x, thresh): return 1 if abs(lin_slope(x)) > float(thresh) else 0

def stats_series(x, alpha, slope_thresh):
    return {
        "mean": float(np.mean(x)),
        "p50": pctl(x, 0.50),
        "p95": pctl(x, 0.95),
        "p99": pctl(x, 0.99),
        "ema_last": ema_last(x, alpha),
        "cv": cv(x),
        "mad": mad(x),
        "slope": lin_slope(x),
        "trend_flag": trend_flag(x, slope_thresh),
    }

def risk_per_metric(stats, wT, wE, wB, wC):
    T = stats["p95"]
    E = stats["ema_last"]
    B = stats["cv"]
    C = float(bool(stats["trend_flag"]))
    return {"T": T, "E": E, "B": B, "C": C, "RISK": wT*T + wE*E + wB*B + wC*C}

# ---- RISK_v2: weighted mean/p95/p50/EMA ----
RISK_V2_W = {
    "w_mean": 0.20,
    "w_p95":  0.30,
    "w_p50":  0.30,  # updated weight
    "w_ema":  0.20,
}

def risk2_per_metric(stats, w=RISK_V2_W):
    mean_v = stats["mean"]
    p95_v  = stats["p95"]
    p50_v  = stats["p50"]
    ema_v  = stats["ema_last"]
    return {
        "mean": mean_v, "p95": p95_v, "p50": p50_v, "ema": ema_v,
        "RISK2": (w["w_mean"]*mean_v + w["w_p95"]*p95_v +
                  w["w_p50"]*p50_v  + w["w_ema"]*ema_v)
    }

def plot_series(t, y, title, outfile):
    plt.figure()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel("time (ticks)"); plt.ylabel("utilization")
    plt.tight_layout(); plt.savefig(outfile, dpi=140); plt.close()

def _row(d, cols): return " | ".join(f"{d[c]:.4f}" for c in cols)

def build_md(params, sS, sO, sD, rS, rO, rD, r2S, r2O, r2D, imgs):
    cols = ["mean","p95","p99","ema_last","cv","mad","slope"]
    md = (
        "# Pattern 2 — Steady Busy\n\n"
        f"**Config:** `N={params['N']}`, `ALPHA={params['ALPHA']:.6f}` (auto-derived=`{params['ALPHA_AUTO']}`)  \n"
        f"Levels (means): SMACT={params['SMACT_MEAN']} • SMOCC={params['SMOCC_MEAN']} • DRAMA={params['DRAMA_MEAN']}  \n"
        f"Noise (std): SMACT={params['SMACT_STD']} • SMOCC={params['SMOCC_STD']} • DRAMA={params['DRAMA_STD']}\n\n"
        "## Plots\n"
        f"![SMACT]({os.path.basename(imgs['smact'])})\n"
        f"![SMOCC]({os.path.basename(imgs['smocc'])})\n"
        f"![DRAMA]({os.path.basename(imgs['drama'])})\n\n"
        "## Window Statistics (per metric)\n"
        "Metric | mean | p95 | p99 | EMA_last | CV | MAD | slope\n"
        "---|---:|---:|---:|---:|---:|---:|---:\n"
        f"SMACT | {_row(sS, cols)}\n"
        f"SMOCC | {_row(sO, cols)}\n"
        f"DRAMA | {_row(sD, cols)}\n\n"
        f"Trend flags: SMACT={sS['trend_flag']} • SMOCC={sO['trend_flag']} • DRAMA={sD['trend_flag']}\n\n"
        "## Per-Metric Risk (no mixing)\n"
        f"Weights: wT={params['wT']}, wE={params['wE']}, wB={params['wB']}, wC={params['wC']}\n\n"
        "|Metric|T (p95)|E (EMA)|B (CV)|C (trend)|RISK|\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"|SMACT|{rS['T']:.4f}|{rS['E']:.4f}|{rS['B']:.4f}|{rS['C']:.1f}|{rS['RISK']:.4f}|\n"
        f"|SMOCC|{rO['T']:.4f}|{rO['E']:.4f}|{rO['B']:.4f}|{rO['C']:.1f}|{rO['RISK']:.4f}|\n"
        f"|DRAMA|{rD['T']:.4f}|{rD['E']:.4f}|{rD['B']:.4f}|{rD['C']:.1f}|{rD['RISK']:.4f}|\n\n"
        "## Per-Metric Risk (v2 only)\n"
        f"Weights: w_mean={RISK_V2_W['w_mean']}, w_p95={RISK_V2_W['w_p95']}, w_p50={RISK_V2_W['w_p50']}, w_ema={RISK_V2_W['w_ema']}\n\n"
        "|Metric|mean|p95|p50|EMA|RISK_v2|\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"|SMACT|{r2S['mean']:.4f}|{r2S['p95']:.4f}|{r2S['p50']:.4f}|{r2S['ema']:.4f}|{r2S['RISK2']:.4f}|\n"
        f"|SMOCC|{r2O['mean']:.4f}|{r2O['p95']:.4f}|{r2O['p50']:.4f}|{r2O['ema']:.4f}|{r2O['RISK2']:.4f}|\n"
        f"|DRAMA|{r2D['mean']:.4f}|{r2D['p95']:.4f}|{r2D['p50']:.4f}|{r2D['ema']:.4f}|{r2D['RISK2']:.4f}|\n"
    )
    return md

def write_readme(path, content, mode="append"):
    if mode == "append" and os.path.exists(path):
        with open(path, "a", encoding="utf-8") as f: f.write("\n\n" + content)
    else:
        with open(path, "w", encoding="utf-8") as f: f.write(content)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    N = int(cfg.get("N", 120))
    seed = int(cfg.get("seed", 42))

    lv = cfg.get("levels", {}) or {}
    smact_mean = float(lv.get("smact_mean", 0.78))
    smocc_mean = float(lv.get("smocc_mean", 0.62))
    drama_mean = float(lv.get("drama_mean", 0.65))

    nz = cfg.get("noise", {}) or {}
    smact_std = float(nz.get("smact_std", 0.03))
    smocc_std = float(nz.get("smocc_std", 0.03))
    drama_std = float(nz.get("drama_std", 0.03))

    alpha_mode = str(cfg.get("alpha_mode", "auto")).lower()
    alpha_cfg = float(cfg.get("alpha", 0.1))
    if alpha_mode == "manual":
        alpha = alpha_cfg; alpha_auto = False
    else:
        alpha = 2.0 / (N + 1.0); alpha_auto = True

    outdir = cfg.get("outdir", ".")
    readme = cfg.get("readme", "README.md")
    readme_mode = str(cfg.get("readme_mode", "append")).lower()

    w = cfg.get("risk_weights", {}) or {}
    wT, wE, wB, wC = float(w.get("wT", 0.5)), float(w.get("wE", 0.3)), float(w.get("wB", 0.1)), float(w.get("wC", 0.1))
    slope_thresh = float(cfg.get("trend_slope_threshold", 0.002))

    np.random.seed(seed); os.makedirs(outdir, exist_ok=True)
    t = np.arange(N)

    smact = np.clip(np.random.normal(loc=smact_mean, scale=smact_std, size=N), 0, 1)
    smocc = np.clip(np.random.normal(loc=smocc_mean, scale=smocc_std, size=N), 0, 1)
    drama = np.clip(np.random.normal(loc=drama_mean, scale=drama_std, size=N), 0, 1)

    s_smact = stats_series(smact, alpha, slope_thresh)
    s_smocc = stats_series(smocc, alpha, slope_thresh)
    s_drama = stats_series(drama, alpha, slope_thresh)

    r_smact = risk_per_metric(s_smact, wT, wE, wB, wC)
    r_smocc = risk_per_metric(s_smocc, wT, wE, wB, wC)
    r_drama = risk_per_metric(s_drama, wT, wE, wB, wC)

    r2_smact = risk2_per_metric(s_smact)
    r2_smocc = risk2_per_metric(s_smocc)
    r2_drama = risk2_per_metric(s_drama)

    img_smact = os.path.join(outdir, "pattern2_smact.png")
    img_smocc = os.path.join(outdir, "pattern2_smocc.png")
    img_drama = os.path.join(outdir, "pattern2_drama.png")
    plot_series(t, smact, "Pattern 2: Steady Busy — SMACT", img_smact)
    plot_series(t, smocc, "Pattern 2: Steady Busy — SMOCC", img_smocc)
    plot_series(t, drama, "Pattern 2: Steady Busy — DRAMA", img_drama)

    section = build_md(
        {"N": N, "ALPHA": alpha, "ALPHA_AUTO": alpha_auto,
         "SMACT_MEAN": smact_mean, "SMOCC_MEAN": smocc_mean, "DRAMA_MEAN": drama_mean,
         "SMACT_STD": smact_std, "SMOCC_STD": smocc_std, "DRAMA_STD": drama_std,
         "wT": wT, "wE": wE, "wB": wB, "wC": wC},
        s_smact, s_smocc, s_drama,
        r_smact, r_smocc, r_drama,
        r2_smact, r2_smocc, r2_drama,
        {"smact": img_smact, "smocc": img_smocc, "drama": img_drama}
    )
    readme_path = os.path.join(outdir, readme)
    write_readme(readme_path, section, mode=readme_mode)

    print(f"alpha = {alpha:.6f} ({'auto' if alpha_auto else 'manual'})")
    print("Per-metric RISK:",
          {"SMACT": round(r_smact["RISK"],6), "SMOCC": round(r_smocc["RISK"],6), "DRAMA": round(r_drama["RISK"],6)})
    print("Per-metric RISK_v2:",
          {"SMACT": round(r2_smact["RISK2"],6), "SMOCC": round(r2_smocc["RISK2"],6), "DRAMA": round(r2_drama["RISK2"],6)})
    print(f"README {readme_mode} to: {readme_path}")

if __name__ == "__main__":
    main()
