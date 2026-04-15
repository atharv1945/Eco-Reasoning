"""
generate_dwt_diagram.py
=======================
Diagram 2: Discrete Wavelet Transform Visualization
Eco-Reasoning Research Paper  |  IEEE Publication Standard

Pipeline:
  1. Load 128-day NASDAQ-100 closing prices
  2. Apply 2-level DWT (Daubechies-4 wavelet)
  3. Reconstruct A2 (trend), D1, D2 (detail / noise) in original price domain
  4. Export DWT_Visualization_EcoReasoning.png @ 300 DPI
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pywt

# ── 0. IEEE-compatible global style ──────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.4,
    "grid.color":         "#cccccc",
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
})

# ── 1. Colour palette ─────────────────────────────────────────────────────────
C_RAW    = "#1a6faf"   # Steel blue   — raw noisy price
C_TREND  = "#2a9d6f"   # Emerald      — A2 approximation (trend)
C_D1     = "#c0392b"   # Crimson      — D1 high-freq detail
C_D2     = "#e67e22"   # Amber        — D2 mid-freq detail
C_NOISE  = "#95a5a6"   # Slate grey   — combined noise fill
C_SHADE  = "#eaf4fb"   # Light blue   — panel background tint

OUTPUT_FILE = "DWT_Visualization_EcoReasoning.png"
WINDOW      = 128          # days
WAVELET     = "db4"        # Daubechies-4
LEVEL       = 2
CSV_NDX     = "ndx_data.csv"


# ── 2. Load data ──────────────────────────────────────────────────────────────
def load_ndx(csv_path: str, window: int) -> tuple[np.ndarray, pd.Series]:
    df = pd.read_csv(csv_path, skiprows=[1, 2])
    df.rename(columns={"Price": "Date"}, inplace=True)
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Pick a visually interesting 128-day window (2022 drawdown → recovery)
    # Find the row closest to 2022-01-01 for a volatile + recovery window
    target = pd.Timestamp("2022-01-03")
    start_idx = (df["Date"] - target).abs().argmin()
    end_idx   = start_idx + window
    if end_idx > len(df):
        start_idx = len(df) - window
        end_idx   = len(df)

    segment = df.iloc[start_idx:end_idx]
    prices  = segment["Close"].values.astype(np.float64)
    dates   = segment["Date"].reset_index(drop=True)
    return prices, dates


# ── 3. DWT decomposition & reconstruction ────────────────────────────────────
def decompose(prices: np.ndarray, wavelet: str = WAVELET, level: int = LEVEL):
    """
    Perform 2-level DWT and reconstruct each sub-band at original length.

    Returns
    -------
    A2_rec  : Level-2 approximation (low-freq trend) in price domain
    D1_rec  : Level-1 detail (high-freq noise)
    D2_rec  : Level-2 detail (mid-freq noise)
    """
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    # coeffs = [cA2, cD2, cD1]  (wavedec orders from coarsest to finest)
    cA2, cD2, cD1 = coeffs

    # Reconstruct each band in isolation (zero out the others)
    n = len(prices)

    # A2 (trend): keep cA2, zero cD2, cD1
    A2_rec = pywt.waverec([cA2, np.zeros_like(cD2), np.zeros_like(cD1)], wavelet)[:n]

    # D2 (mid detail): zero cA2 & cD1, keep cD2
    D2_rec = pywt.waverec([np.zeros_like(cA2), cD2, np.zeros_like(cD1)], wavelet)[:n]

    # D1 (fine detail): zero cA2 & cD2, keep cD1
    D1_rec = pywt.waverec([np.zeros_like(cA2), np.zeros_like(cD2), cD1], wavelet)[:n]

    # Sanity: A2 + D2 + D1 ≈ original (perfect reconstruction)
    residual = np.max(np.abs((A2_rec + D2_rec + D1_rec) - prices))
    print(f"  Max reconstruction residual : {residual:.6f}  (should be ≈ 0)")

    return A2_rec, D1_rec, D2_rec


# ── 4. Plotting ───────────────────────────────────────────────────────────────
def build_figure(prices, dates, A2, D1, D2, wavelet=WAVELET, level=LEVEL):
    t = np.arange(len(prices))   # integer x-axis (days)

    # ── Figure layout: 3 rows with different height ratios ───────────────────
    fig = plt.figure(figsize=(10, 8.5))
    fig.suptitle(
        "Discrete Wavelet Transform (DWT) Decomposition — NASDAQ-100\n"
        r"Wavelet: Daubechies-4 ($db4$) · Level 2 · 128-Day Window",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[2.2, 2.2, 1.8],
        hspace=0.52,
    )

    ax0 = fig.add_subplot(gs[0])   # Raw price
    ax1 = fig.add_subplot(gs[1])   # A2 trend
    ax2 = fig.add_subplot(gs[2])   # D1 + D2 details

    # ── Shared x-axis tick labels (quarterly marks) ──────────────────────────
    tick_step = 16
    tick_locs = np.arange(0, len(t), tick_step)
    tick_lbls = [str(dates.iloc[i].date()) if i < len(dates) else ""
                 for i in tick_locs]

    def style_ax(ax, ylabel, title, color=None):
        ax.set_facecolor(C_SHADE)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_title(title, loc="left", pad=6, fontsize=11, fontweight="bold")
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_lbls, rotation=30, ha="right", fontsize=9)
        ax.set_xlabel("Time $t$ (trading days)", labelpad=4)
        if color:
            for spine in ax.spines.values():
                if spine.get_visible():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.4)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1 — Raw Price Signal
    # ─────────────────────────────────────────────────────────────────────────
    ax0.plot(t, prices, color=C_RAW, linewidth=1.3, label="Raw Price $P_t$", zorder=3)
    ax0.fill_between(t, prices.min() * 0.998, prices,
                     color=C_RAW, alpha=0.08, zorder=2)

    # Annotate peak and trough
    peak_i  = int(np.argmax(prices))
    trough_i = int(np.argmin(prices))
    ax0.annotate(
        f"Peak\n${prices[peak_i]:,.0f}",
        xy=(peak_i, prices[peak_i]),
        xytext=(peak_i + 6, prices[peak_i] + (prices.max() - prices.min()) * 0.04),
        fontsize=8.5, color=C_RAW, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_RAW, lw=1.0),
    )
    ax0.annotate(
        f"Trough\n${prices[trough_i]:,.0f}",
        xy=(trough_i, prices[trough_i]),
        xytext=(trough_i + 5, prices[trough_i] - (prices.max() - prices.min()) * 0.07),
        fontsize=8.5, color="#555555", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
    )

    style_ax(ax0, "Price (USD)", "Panel 1 — Raw Price Signal  [Noisy Input $P_t$]", C_RAW)
    ax0.legend(loc="upper right", framealpha=0.9)
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2 — A2 Level-2 Approximation (Trend)
    # ─────────────────────────────────────────────────────────────────────────
    # Ghost raw signal for reference
    ax1.plot(t, prices, color=C_RAW, linewidth=0.7, alpha=0.30,
             linestyle="--", label="Raw $P_t$ (reference)", zorder=2)
    ax1.plot(t, A2, color=C_TREND, linewidth=2.0,
             label=r"$A_2$ — Level-2 Approximation (Trend)", zorder=3)
    ax1.fill_between(t, A2.min() * 0.998, A2,
                     color=C_TREND, alpha=0.10, zorder=2)

    # Label smoothing region
    mid = len(t) // 2
    ax1.annotate(
        "Denoised trend\nfed to LightGBM",
        xy=(mid, A2[mid]),
        xytext=(mid - 20, A2[mid] + (A2.max() - A2.min()) * 0.25),
        fontsize=8.5, color=C_TREND, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_TREND, lw=1.0),
    )

    style_ax(ax1,
             "Price (USD)",
             r"Panel 2 — $A_2$ Approximation  [Low-Frequency Trend · System 1 Input]",
             C_TREND)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3 — D1 & D2 Detail Coefficients (Noise)
    # ─────────────────────────────────────────────────────────────────────────
    ax2.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":", zorder=1)

    # Fill-between for D1 (high-freq)
    ax2.fill_between(t, 0, D1, where=(D1 >= 0), color=C_D1, alpha=0.45, label=None)
    ax2.fill_between(t, 0, D1, where=(D1 <  0), color=C_D1, alpha=0.45, label=None)
    ax2.plot(t, D1, color=C_D1, linewidth=0.9,
             label=r"$D_1$ — Level-1 Detail (High-Freq Noise)", zorder=3)

    # D2 on top
    ax2.plot(t, D2, color=C_D2, linewidth=1.3, linestyle="-.",
             label=r"$D_2$ — Level-2 Detail (Mid-Freq Noise)", zorder=4)

    # Shade total noise band
    combined_noise = D1 + D2
    ax2.fill_between(t, 0, combined_noise,
                     color=C_NOISE, alpha=0.12,
                     label="Combined noise $D_1+D_2$ (discarded)", zorder=2)

    style_ax(ax2,
             "Amplitude (USD)",
             r"Panel 3 — Detail Coefficients  [$D_1$, $D_2$ · High/Mid-Frequency Noise Discarded]",
             C_D1)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:+,.0f}"))

    # ─────────────────────────────────────────────────────────────────────────
    # Shared caption / annotation box
    # ─────────────────────────────────────────────────────────────────────────
    caption = (
        r"$\bf{DWT\ Signal\ Flow:}$  "
        r"$P_t \to [A_2 + D_2 + D_1]$ (perfect reconstruction)  |  "
        "System 1 retains "
        r"$A_2$ for LightGBM regression  |  "
        r"$D_1, D_2$ discarded (market microstructure noise)"
    )
    fig.text(
        0.5, 0.005, caption,
        ha="center", va="bottom",
        fontsize=8.5, style="italic", color="#444444",
    )

    return fig


# ── 5. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DWT Visualization — Eco-Reasoning Research Paper")
    print("=" * 60)

    print(f"\n  [1/4] Loading NASDAQ-100 data ({WINDOW}-day window) ...")
    prices, dates = load_ndx(CSV_NDX, WINDOW)
    print(f"        Window : {dates.iloc[0].date()}  →  {dates.iloc[-1].date()}")
    print(f"        Price range : ${prices.min():,.2f}  –  ${prices.max():,.2f}")

    print(f"\n  [2/4] Applying {LEVEL}-level DWT (wavelet={WAVELET}) ...")
    A2, D1, D2 = decompose(prices)
    print(f"        A2 range : ${A2.min():,.2f}  –  ${A2.max():,.2f}")
    print(f"        D1 std   : ${D1.std():,.4f}")
    print(f"        D2 std   : ${D2.std():,.4f}")

    print(f"\n  [3/4] Rendering figure ...")
    fig = build_figure(prices, dates, A2, D1, D2)

    print(f"\n  [4/4] Saving → {OUTPUT_FILE}  (300 DPI) ...")
    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print(f"        OK — file written: {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("  DIAGRAM GENERATION COMPLETE")
    print("=" * 60 + "\n")
