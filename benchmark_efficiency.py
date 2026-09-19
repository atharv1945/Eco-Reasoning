"""
benchmark_efficiency.py — Eco-Reasoning vs Standard RAG/LLM Efficiency Benchmark
==================================================================================
Simulates 10,000 market ticks (95% Normal / 5% Anomaly) and compares:
  - Standard RAG/LLM  : every tick hits the LLM  (800 ms, $0.001 / tick)
  - Eco-Reasoning v2  : Path A = 0.55 ms, Path B = 800 ms, cost only on Path B

Outputs: efficiency_benchmark.png  (professional dark-themed financial chart)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ============================================================================
# Simulation Parameters
# ============================================================================

TOTAL_TICKS       = 10_000
NORMAL_FRACTION   = 0.95         # Path A
ANOMALY_FRACTION  = 0.05         # Path B

STANDARD_LATENCY_MS   = 800.0    # ms per tick
STANDARD_COST_PER_TICK = 0.001   # USD per tick

ECO_PATH_A_LATENCY_MS = 0.55     # ms (LightGBM + Wavelet, from benchmarks)
ECO_PATH_B_LATENCY_MS = 800.0    # ms (LLM, same as standard)
ECO_COST_PER_TICK_B   = 0.001    # USD (only for Path B)

# ============================================================================
# Calculate Metrics
# ============================================================================

normal_ticks  = int(TOTAL_TICKS * NORMAL_FRACTION)
anomaly_ticks = int(TOTAL_TICKS * ANOMALY_FRACTION)

# — Standard System —
std_total_latency_s = (TOTAL_TICKS * STANDARD_LATENCY_MS) / 1_000      # seconds
std_total_cost      = TOTAL_TICKS * STANDARD_COST_PER_TICK              # USD

# — Eco-Reasoning System —
eco_latency_path_a_s = (normal_ticks  * ECO_PATH_A_LATENCY_MS) / 1_000
eco_latency_path_b_s = (anomaly_ticks * ECO_PATH_B_LATENCY_MS) / 1_000
eco_total_latency_s  = eco_latency_path_a_s + eco_latency_path_b_s
eco_total_cost       = anomaly_ticks * ECO_COST_PER_TICK_B

# — Savings —
latency_reduction_pct = (1 - eco_total_latency_s / std_total_latency_s) * 100
cost_reduction_pct    = (1 - eco_total_cost      / std_total_cost)      * 100

print("=" * 60)
print("ECO-REASONING EFFICIENCY BENCHMARK")
print("=" * 60)
print(f"  Ticks           : {TOTAL_TICKS:,}")
print(f"  Normal (Path A) : {normal_ticks:,}  ({NORMAL_FRACTION*100:.0f}%)")
print(f"  Anomaly (Path B): {anomaly_ticks:,}  ({ANOMALY_FRACTION*100:.0f}%)")
print()
print(f"  Standard Latency: {std_total_latency_s:,.1f} s")
print(f"  Eco     Latency : {eco_total_latency_s:,.1f} s  ({latency_reduction_pct:.1f}% savings)")
print()
print(f"  Standard Cost   : ${std_total_cost:,.2f}")
print(f"  Eco     Cost    : ${eco_total_cost:,.2f}  ({cost_reduction_pct:.1f}% savings)")
print("=" * 60)

# ============================================================================
# Dark-Theme Palette
# ============================================================================

BG_DARK    = "#0D1117"
BG_PANEL   = "#161B22"
ACCENT_STD = "#FF4C4C"   # red  → standard (expensive)
ACCENT_ECO = "#00C853"   # green → eco (efficient)
TEXT_DIM   = "#8B949E"
TEXT_BRIGHT= "#E6EDF3"
GRID_COLOR = "#21262D"
GOLD       = "#F9A825"

# ============================================================================
# Figure Layout
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor(BG_DARK)

for ax in axes:
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_DIM, labelsize=11)
    ax.spines[:].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)

BAR_W   = 0.42
X       = np.array([0.0, 1.0])
LABELS  = ["Standard\nRAG/LLM", "Eco-Reasoning\nv2"]

# ── helper to add value label above each bar ─────────────────────────────────
def _label(ax, rect, txt, color=TEXT_BRIGHT):
    h = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        h * 1.015,
        txt,
        ha="center", va="bottom",
        color=color, fontsize=12, fontweight="bold",
    )

# ── savings annotation ────────────────────────────────────────────────────────
def _savings_arrow(ax, x0, x1, y_top, pct, unit=""):
    mid_x = (x0 + x1) / 2
    ax.annotate(
        "",
        xy=(x1, y_top * 0.92), xytext=(x0, y_top * 0.92),
        arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.8),
    )
    ax.text(
        mid_x, y_top * 0.97,
        f"−{pct:.1f}%{unit}",
        ha="center", va="bottom",
        color=GOLD, fontsize=13, fontweight="bold",
    )

# ============================================================================
# Chart 1 — Total Cumulative Latency
# ============================================================================

ax1 = axes[0]
vals_lat = [std_total_latency_s, eco_total_latency_s]
colors_lat = [ACCENT_STD, ACCENT_ECO]

bars_lat = []
for xi, (v, c) in enumerate(zip(vals_lat, colors_lat)):
    b = ax1.bar(xi, v, width=BAR_W, color=c, zorder=3,
                linewidth=1.2, edgecolor=BG_DARK)
    bars_lat.append(b[0])

ax1.set_xticks(X)
ax1.set_xticklabels(LABELS, color=TEXT_BRIGHT, fontsize=12)
ax1.set_xlim(-0.55, 1.55)
ax1.set_ylim(0, std_total_latency_s * 1.22)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f} s"))
ax1.set_ylabel("Cumulative Latency (seconds)", color=TEXT_DIM, fontsize=12)
ax1.set_title("Total Cumulative Latency\n10,000 Ticks", color=TEXT_BRIGHT,
              fontsize=14, fontweight="bold", pad=14)

_label(ax1, bars_lat[0], f"{std_total_latency_s:,.0f} s", ACCENT_STD)
_label(ax1, bars_lat[1], f"{eco_total_latency_s:,.1f} s", ACCENT_ECO)

_savings_arrow(ax1, 0, 1, std_total_latency_s * 1.15, latency_reduction_pct)

# Path B breakdown annotation on eco bar
ax1.text(
    1, eco_total_latency_s * 0.5,
    f"Path A: {eco_latency_path_a_s:.2f}s\nPath B: {eco_latency_path_b_s:.1f}s",
    ha="center", va="center",
    color=BG_DARK, fontsize=9, fontweight="bold",
)

# ============================================================================
# Chart 2 — Total API Cost
# ============================================================================

ax2 = axes[1]
vals_cost = [std_total_cost, eco_total_cost]
colors_cost = [ACCENT_STD, ACCENT_ECO]

bars_cost = []
for xi, (v, c) in enumerate(zip(vals_cost, colors_cost)):
    b = ax2.bar(xi, v, width=BAR_W, color=c, zorder=3,
                linewidth=1.2, edgecolor=BG_DARK)
    bars_cost.append(b[0])

ax2.set_xticks(X)
ax2.set_xticklabels(LABELS, color=TEXT_BRIGHT, fontsize=12)
ax2.set_xlim(-0.55, 1.55)
ax2.set_ylim(0, std_total_cost * 1.22)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax2.set_ylabel("Total API Cost (USD)", color=TEXT_DIM, fontsize=12)
ax2.set_title("Total API Cost\n10,000 Ticks", color=TEXT_BRIGHT,
              fontsize=14, fontweight="bold", pad=14)

_label(ax2, bars_cost[0], f"${std_total_cost:,.2f}", ACCENT_STD)
_label(ax2, bars_cost[1], f"${eco_total_cost:,.2f}", ACCENT_ECO)

_savings_arrow(ax2, 0, 1, std_total_cost * 1.15, cost_reduction_pct)

ax2.text(
    1, eco_total_cost * 0.5,
    f"{anomaly_ticks:,} LLM calls\n({ANOMALY_FRACTION*100:.0f}% of ticks)",
    ha="center", va="center",
    color=BG_DARK, fontsize=9, fontweight="bold",
)

# ============================================================================
# Figure-level annotations
# ============================================================================

fig.suptitle(
    "Eco-Reasoning v2 — Efficiency vs Standard RAG/LLM",
    color=TEXT_BRIGHT, fontsize=17, fontweight="bold", y=1.01,
)

# Subtitle / simulation config
fig.text(
    0.5, 0.97,
    f"Simulation: {TOTAL_TICKS:,} ticks  |  "
    f"{NORMAL_FRACTION*100:.0f}% Normal (Path A · {ECO_PATH_A_LATENCY_MS} ms)  |  "
    f"{ANOMALY_FRACTION*100:.0f}% Anomaly (Path B · {ECO_PATH_B_LATENCY_MS:.0f} ms)",
    ha="center", color=TEXT_DIM, fontsize=10,
)

# Legend
legend_patches = [
    mpatches.Patch(color=ACCENT_STD, label="Standard RAG/LLM  — all ticks use LLM"),
    mpatches.Patch(color=ACCENT_ECO, label="Eco-Reasoning v2   — LLM only on anomalies"),
    mpatches.Patch(color=GOLD,       label="Savings (Eco vs Standard)"),
]
fig.legend(
    handles=legend_patches,
    loc="lower center", ncol=3,
    frameon=True, framealpha=0.15,
    facecolor=BG_PANEL, edgecolor=GRID_COLOR,
    labelcolor=TEXT_BRIGHT, fontsize=10,
    bbox_to_anchor=(0.5, -0.05),
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# ============================================================================
# Save
# ============================================================================

OUT = "efficiency_benchmark.png"
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=BG_DARK)
print(f"\n✅  Saved → {OUT}")
plt.close()
