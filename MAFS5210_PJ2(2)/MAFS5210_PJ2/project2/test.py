from pathlib import Path

import matplotlib.pyplot as plt
from project2.data import load_market_data
from project2.barra_proxy import build_barra_style_factor_returns
from project2.risk_model import fit_time_series_factor_model

project_root = Path("/Users/henrysmacbook/Desktop/ipynb/MAFS5210_PJ2_v3")
project2_data_dir = project_root / "Project2_data"
output_dir = project_root / "Project2_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

market_data = load_market_data(project_root / "Project1")
factor_returns = build_barra_style_factor_returns(
    market_data=market_data,
    project2_data_dir=project2_data_dir,
    top_bottom_quantile=0.2,
)

csv_path = output_dir / "barra_style_factor_returns.csv"
factor_returns.to_csv(csv_path)
print(f"Saved: {csv_path}")

cum_returns = (1.0 + factor_returns).cumprod() - 1.0

sector_cols = [c for c in cum_returns.columns if c.startswith("Sector_")]
style_cols = [c for c in cum_returns.columns if c not in sector_cols and c != "Market"]

# Sector factors
ax1 = cum_returns[sector_cols].plot(
    figsize=(12, 6),
    linewidth=1.5,
    colormap="tab20",
)
ax1.set_title("Sector Factor Cumulative Returns")
ax1.set_xlabel("Date")
ax1.set_ylabel("Cumulative Return")
ax1.grid(True, alpha=0.3)
plt.tight_layout()

sector_png = output_dir / "sector_factor_cumulative_returns.png"
plt.savefig(sector_png, dpi=200)
print(f"Saved: {sector_png}")
plt.close()

# Style factors
ax2 = cum_returns[style_cols].plot(
    figsize=(12, 6),
    linewidth=1.5,
    colormap="tab20",
)
ax2.set_title("Style Factor Cumulative Returns")
ax2.set_xlabel("Date")
ax2.set_ylabel("Cumulative Return")
ax2.grid(True, alpha=0.3)
plt.tight_layout()

style_png = output_dir / "style_factor_cumulative_returns.png"
plt.savefig(style_png, dpi=200)
print(f"Saved: {style_png}")
plt.close()

# Build inputs for risk model fit
stock_returns = market_data.returns
raw_factor_returns = factor_returns
proxy_benchmark_returns = market_data.benchmark["ben_daily_return"]

aligned_index = (
    stock_returns.index.intersection(raw_factor_returns.index).intersection(proxy_benchmark_returns.index)
)
factor_returns_pre_orth = raw_factor_returns.loc[aligned_index].dropna()
proxy_window = proxy_benchmark_returns.loc[factor_returns_pre_orth.index].dropna()
factor_returns_pre_orth = factor_returns_pre_orth.loc[proxy_window.index]

factor_corr_pre_orth = factor_returns_pre_orth.corr()
factor_corr_pre_orth_path = output_dir / "factor_correlation_pre_orth.csv"
factor_corr_pre_orth.to_csv(factor_corr_pre_orth_path)
print(f"Saved: {factor_corr_pre_orth_path}")

fit = fit_time_series_factor_model(
    stock_returns=stock_returns,
    raw_factor_returns=raw_factor_returns,
    proxy_benchmark_returns=proxy_benchmark_returns,
    proxy_benchmark_weights=None,
)

factor_corr_post_orth = fit.factor_returns.corr()
factor_corr_post_orth_path = output_dir / "factor_correlation_post_orth.csv"
factor_corr_post_orth.to_csv(factor_corr_post_orth_path)
print(f"Saved: {factor_corr_post_orth_path}")

# Plot and save heatmap of the full post-orth correlation matrix with annotations
fig, ax = plt.subplots(figsize=(10, 10))
data = factor_corr_post_orth.values
im = ax.imshow(data, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
ax.set_xticks(range(len(factor_corr_post_orth.columns)))
ax.set_yticks(range(len(factor_corr_post_orth.index)))
ax.set_xticklabels(factor_corr_post_orth.columns, rotation=90, fontsize=8)
ax.set_yticklabels(factor_corr_post_orth.index, fontsize=8)
ax.set_title("Post-Orthogonalization Factor Correlation")

# Annotate each cell with the correlation value
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
heatmap_png = output_dir / "factor_correlation_post_orth_heatmap.png"
plt.savefig(heatmap_png, dpi=300)
print(f"Saved: {heatmap_png}")
plt.close()
factor_cov = fit.factor_cov
print("Factor covariance shape:", factor_cov.shape)
print(factor_cov.head())

out_path = output_dir / "factor_covariance.csv"
factor_cov.to_csv(out_path)
print(f"Saved: {out_path}")