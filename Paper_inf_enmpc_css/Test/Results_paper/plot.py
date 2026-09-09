"""Plot NMPC results with y-axes shared across panels.

Every panel in a row uses identical y-limits so the controllers can be compared
by eye without the axis ranges exaggerating the differences.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Configuration
HERE = Path(__file__).resolve().parent
# Keep Times New Roman when available, with installed serif alternatives on Linux.
_available_fonts = {font.name for font in font_manager.fontManager.ttflist}
FONT_FAMILY = next(
    name for name in (
        "Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"
    ) if name in _available_fonts
)
X_MAX_HOURS = 60
OCSS_FILE = "css.xlsx"
ENMPC_RUNS = [
    ("finite-1-cycle_tracking.xlsx", "Tracking"),
    ("finite-1-cycle.xlsx", "1 cycle"),
    ("finite-10-cycle.xlsx", "10-cycle eNMPC"),
    ("Inf.xlsx", "Inf eNMPC"),
]
TIME_RUNS = [
    ("finite-1-cycle.xlsx", "1-cycle eNMPC"),
    ("finite-1-cycle_tracking.xlsx", "1-cycle tracking"),
    ("finite-10-cycle.xlsx", "10-cycle eNMPC"),
    ("finite-10-cycle_tracking.xlsx", "10-cycle tracking"),
    ("Inf.xlsx", "Inf eNMPC"),
    ("Inf_tracking.xlsx", "Inf tracking"),
]
COST_RUNS = [
    ("finite-1-cycle_tracking.xlsx", "1-cycle tracking"),
    ("finite-1-cycle.xlsx", "1-cycle eNMPC"),
    ("Inf.xlsx", "Inf/10-cycle eNMPC"),
]
OCSS_CYCLE_HOURS = 6
OCSS_DROP_LAST = 1
FLOW_SOURCES = ["source_1"]
PRESSURE_SOURCES = []
FLOW_SCALE = 1.0
PRESSURE_SCALE = 1.0
# Power conversion used for both plots and reported energy.
POWER_SCALE = 1000.0
OUTPUT_PNG = "enmpc_comparison.png"
OUTPUT_TIME_PNG = "mpc_step_times.png"
OUTPUT_LYAPUNOV_PNG = "lyapunov.png"
OUTPUT_COST_PNG = "total_cost.png"
OUTPUT_COST_6H_PNG = "cost_6h.png"
OUTPUT_COST_12H_PNG = "cost_12h.png"
OUTPUT_TXT = "results.txt"
DPI = 200

# Shared-axis settings.
Y_PAD_FRAC = 0.05          # fraction of the data span added above and below
HIDE_INNER_TICKLABELS = True  # only the outer columns carry tick labels
# Hand-set limits win over the auto-computed shared range.  Use this when the
# startup transient of one run squashes the steady-state cycles of the others,
# e.g. Y_LIMITS_OVERRIDE = {"flow": (130, 200)}.
Y_LIMITS_OVERRIDE = {}     # keys: "flow", "pressure", "power"


def select_existing_runs(folder, candidates):
    """Return only the explicitly requested workbooks, in requested order."""
    runs = [(fn, label) for fn, label in candidates if (folder / fn).exists()]
    if not runs:
        raise FileNotFoundError(f"No requested run Excel files found in {folder}")
    return runs


RUN_FILES = select_existing_runs(HERE, ENMPC_RUNS)
TIME_FILES = select_existing_runs(HERE, TIME_RUNS)


def compute_cost_kwh(pwr_df, pwr_cols, dt=3600, power_to_kw=1000):
    """Total compressor cost, including the final row."""
    total_power = float(pwr_df[pwr_cols].values.sum())
    return total_power * (dt / 3600.0) * power_to_kw


def compute_cost_through_hour(
        pwr_df, pwr_cols, end_hour=60, dt=3600, power_to_kw=1000):
    """Sum compressor cost at integer hours 0 through ``end_hour`` inclusive."""
    n_required = int(end_hour) + 1
    if len(pwr_df) < n_required:
        return None
    block = pwr_df[pwr_cols].iloc[:n_required].values
    return float(block.sum()) * (dt / 3600.0) * power_to_kw


def load_run(path):
    """Load the sheets used by the plots and cost calculation."""
    src_w = pd.read_excel(path, sheet_name="wSource")
    src_p = pd.read_excel(path, sheet_name="pSource")
    pwr = pd.read_excel(path, sheet_name="compressor power")
    for df in (src_w, src_p, pwr):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
    return src_w, src_p, pwr


def sort_by_trailing_int(cols):
    """Sort variable columns by the trailing integer in the entity name."""
    def key(c):
        try:
            return int(c.split("'")[1].split("_")[-1])
        except Exception:
            return 0
    return sorted(cols, key=key)


def entity_name(col):
    """Extract 'source_1' from \"wSource['source_1', :]\" etc."""
    return col.split("'")[1]


def truncate_to_hours(src_w, src_p, pwr, max_hours):
    """Keep rows 0 through max_hours, including the endpoint."""
    keep = min(max_hours + 1, len(src_w))
    return (src_w.iloc[:keep].reset_index(drop=True),
            src_p.iloc[:keep].reset_index(drop=True),
            pwr.iloc[:keep].reset_index(drop=True))


def tile_ocss(ocss_t, ocss_vals, target_t, cycle_hours):
    """Periodic interpolation: value at t+cycle_hours equals value at t."""
    phase = np.mod(target_t, cycle_hours)
    t_ext = np.concatenate([ocss_t, [cycle_hours]])
    v_ext = np.concatenate([ocss_vals, [ocss_vals[0]]])
    return np.interp(phase, t_ext, v_ext)


def shared_limits(value_arrays, pad_frac=Y_PAD_FRAC):
    """Common (lo, hi) covering every array, padded by ``pad_frac``."""
    finite = []
    for arr in value_arrays:
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            finite.append(a)
    if not finite:
        return None
    stacked = np.concatenate(finite)
    lo, hi = float(stacked.min()), float(stacked.max())
    span = hi - lo
    if span <= 0:
        span = abs(hi) or 1.0
    return lo - pad_frac * span, hi + pad_frac * span


def load_cpu_time_seconds(path):
    """Read total CPU time from runtime, or fall back to timing."""
    xl = pd.ExcelFile(path)
    if "runtime" in xl.sheet_names:
        df = pd.read_excel(path, sheet_name="runtime")
        if {"iteration", "iteration_time_s"}.issubset(df.columns) and len(df) > 0:
            iterations = pd.to_numeric(df["iteration"], errors="coerce")
            vals = pd.to_numeric(df["iteration_time_s"], errors="coerce")
            vals = vals[
                (iterations >= 0) & (iterations < X_MAX_HOURS)
            ].dropna()
            if len(vals) > 0:
                return float(vals.sum())
    if "timing" in xl.sheet_names:
        df = pd.read_excel(path, sheet_name="timing")
        if "total_runtime_s" in df.columns:
            val = df["total_runtime_s"].iloc[0]
            if pd.notna(val):
                return float(val)
    return None


def load_first_last_iter_seconds(path):
    """Return the first and last MPC solve times."""
    xl = pd.ExcelFile(path)
    if "runtime" not in xl.sheet_names:
        return None, None
    df = pd.read_excel(path, sheet_name="runtime")
    if not {"iteration", "iteration_time_s"}.issubset(df.columns):
        return None, None
    iterations = pd.to_numeric(df["iteration"], errors="coerce")
    vals = pd.to_numeric(df["iteration_time_s"], errors="coerce")
    mask = (
        (iterations >= 0) & (iterations < X_MAX_HOURS) & vals.notna()
    )
    sub = df[mask]
    if len(sub) == 0:
        return None, None
    it = pd.to_numeric(sub["iteration"], errors="coerce")
    first = float(sub["iteration_time_s"].iloc[it.argmin()])
    last = float(sub["iteration_time_s"].iloc[it.argmax()])
    return first, last


def load_iter_series(path):
    """Return solve times for steps 0..X_MAX_HOURS-1."""
    xl = pd.ExcelFile(path)
    if "runtime" not in xl.sheet_names:
        return None, None
    df = pd.read_excel(path, sheet_name="runtime")
    if not {"iteration", "iteration_time_s"}.issubset(df.columns):
        return None, None
    iterations = pd.to_numeric(df["iteration"], errors="coerce")
    vals = pd.to_numeric(df["iteration_time_s"], errors="coerce")
    mask = (
        (iterations >= 0) & (iterations < X_MAX_HOURS) & vals.notna()
    )
    sub = df[mask].copy()
    sub = sub.sort_values("iteration")
    return sub["iteration"].to_numpy(), sub["iteration_time_s"].to_numpy()


def load_lyapunov_series(path):
    xl = pd.ExcelFile(path)
    if "lyapunov" not in xl.sheet_names:
        return None

    df = pd.read_excel(path, sheet_name="lyapunov")
    if "t [h]" not in df.columns:
        return None
    if {"V_finite", "V_infinite", "V_total"}.issubset(df.columns):
        return df[["t [h]", "V_finite", "V_infinite", "V_total"]]

    log_path = path.with_suffix(".txt")
    if not log_path.exists():
        return None
    pattern = re.compile(
        r"V: finite=([+\-\d.eE]+), infinite=([+\-\d.eE]+), "
        r"total=([+\-\d.eE]+)"
    )
    values = [
        tuple(map(float, match.groups()))
        for match in pattern.finditer(log_path.read_text(errors="replace"))
    ]
    if not values:
        return None

    count = min(len(df), len(values))
    result = pd.DataFrame(values[:count], columns=[
        "V_finite", "V_infinite", "V_total"
    ])
    result.insert(0, "t [h]", df["t [h]"].iloc[:count].to_numpy())
    if "V" in df.columns:
        result["V_total"] = df["V"].iloc[:count].to_numpy()
    return result


def format_cpu_time(seconds):
    return f"{seconds:.2f} s"


# Load data
runs = []
for fn, title in RUN_FILES:
    src_w, src_p, pwr = load_run(HERE / fn)
    src_w, src_p, pwr = truncate_to_hours(
        src_w, src_p, pwr, X_MAX_HOURS
    )
    cpu_seconds = load_cpu_time_seconds(HERE / fn)
    t = np.arange(len(src_w))
    runs.append((title, t, src_w, src_p, pwr, cpu_seconds))

ocss_src_w, ocss_src_p, ocss_pwr = load_run(HERE / OCSS_FILE)
keep = min(OCSS_CYCLE_HOURS, len(ocss_src_w) - OCSS_DROP_LAST)
ocss_src_w = ocss_src_w.iloc[:keep].reset_index(drop=True)
ocss_src_p = ocss_src_p.iloc[:keep].reset_index(drop=True)
ocss_pwr = ocss_pwr.iloc[:keep].reset_index(drop=True)
ocss_t = np.arange(len(ocss_src_w))

wS_cols = sort_by_trailing_int([c for c in ocss_src_w.columns if "wSource" in c])
pS_cols = sort_by_trailing_int([c for c in ocss_src_p.columns if "pSource" in c])
pwr_cols = sort_by_trailing_int([c for c in ocss_pwr.columns if "compressor_P" in c])
wS_by_src = {entity_name(c): c for c in wS_cols}
pS_by_src = {entity_name(c): c for c in pS_cols}
cmap10 = plt.get_cmap("tab10")
all_sources = FLOW_SOURCES + PRESSURE_SOURCES
src_color = {name: cmap10(i % 10) for i, name in enumerate(all_sources)}
pwr_color = {c: cmap10(i % 10) for i, c in enumerate(pwr_cols)}


# Shared y-limits: collect every series that any panel will draw.
flow_values, pressure_values, power_values = [], [], []
for _, t, src_w, src_p, pwr, _ in runs:
    for name in FLOW_SOURCES:
        c = wS_by_src.get(name)
        if c is not None and c in src_w.columns:
            flow_values.append(src_w[c].values * FLOW_SCALE)
            flow_values.append(tile_ocss(
                ocss_t, ocss_src_w[c].values * FLOW_SCALE, t, OCSS_CYCLE_HOURS))
    for name in PRESSURE_SOURCES:
        c = pS_by_src.get(name)
        if c is not None and c in src_p.columns:
            pressure_values.append(src_p[c].values * PRESSURE_SCALE)
            pressure_values.append(tile_ocss(
                ocss_t, ocss_src_p[c].values * PRESSURE_SCALE, t,
                OCSS_CYCLE_HOURS))
    for c in pwr_cols:
        if c in pwr.columns:
            power_values.append(pwr[c].values * POWER_SCALE)
        if c in ocss_pwr.columns:
            power_values.append(tile_ocss(
                ocss_t, ocss_pwr[c].values * POWER_SCALE, t, OCSS_CYCLE_HOURS))

FLOW_YLIM = Y_LIMITS_OVERRIDE.get("flow", shared_limits(flow_values))
PRESSURE_YLIM = Y_LIMITS_OVERRIDE.get("pressure", shared_limits(pressure_values))
POWER_YLIM = Y_LIMITS_OVERRIDE.get("power", shared_limits(power_values))


# NMPC comparison
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 11,
    "axes.grid": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "mathtext.cal": FONT_FAMILY,
})


def add_panel_label(ax, idx):
    ax.text(
        -0.08, 1.04, f"({chr(ord('a') + idx)})",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        va="bottom", ha="left",
    )

n_cols = len(runs)
fig, axes = plt.subplots(2, n_cols, figsize=(5.3 * n_cols, 8),
                         constrained_layout=True)
if n_cols == 1:
    axes = axes.reshape(2, 1)

for col_idx, (title, t, src_w, src_p, pwr, _) in enumerate(runs):
    ax_top = axes[0, col_idx]
    ax_bot = axes[1, col_idx]

    handles, labels_ = [], []

    for name in FLOW_SOURCES:
        if name in wS_by_src and wS_by_src[name] in src_w.columns:
            c = wS_by_src[name]
            color = src_color[name]
            line, = ax_top.plot(t, src_w[c].values * FLOW_SCALE,
                                color=color, lw=1.7, label=name)
            handles.append(line)
            labels_.append(name.replace("source_", "Source"))
            y_ocss = tile_ocss(ocss_t, ocss_src_w[c].values * FLOW_SCALE,
                               t, OCSS_CYCLE_HOURS)
            ax_top.plot(t, y_ocss, color=color, lw=1.1, linestyle=":", alpha=0.75)

    ocss_proxy = plt.Line2D([], [], color="black", linestyle=":", lw=1.2)
    handles.append(ocss_proxy)
    labels_.append("OCSS")

    add_panel_label(ax_top, col_idx)
    add_panel_label(ax_bot, col_idx + n_cols)
    ax_top.set_xlabel("Time (h)")
    ax_top.set_ylabel("Flow (kg/s)")
    ax_top.set_xlim(0, X_MAX_HOURS)
    if FLOW_YLIM is not None:
        ax_top.set_ylim(*FLOW_YLIM)
    ax_top.legend(handles, labels_, loc="lower right",
                  fontsize=8, framealpha=0.9)

    for c in pwr_cols:
        if c in pwr.columns:
            label = entity_name(c).replace("compressorStation_", "C")
            ax_bot.plot(t, pwr[c].values * POWER_SCALE,
                        color=pwr_color[c], lw=1.7, label=label)
    for c in pwr_cols:
        if c in ocss_pwr.columns:
            y_ocss = tile_ocss(ocss_t, ocss_pwr[c].values * POWER_SCALE,
                               t, OCSS_CYCLE_HOURS)
            ax_bot.plot(t, y_ocss, color=pwr_color[c],
                        lw=1.1, linestyle=":", alpha=0.75)
    ax_bot.plot([], [], color="black", linestyle=":", lw=1.2, label="OCSS")

    ax_bot.set_xlabel("Time (h)")
    ax_bot.set_ylabel("Energy (kWh)")
    ax_bot.set_xlim(0, X_MAX_HOURS)
    if POWER_YLIM is not None:
        ax_bot.set_ylim(*POWER_YLIM)
    ax_bot.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

    # With identical limits the inner tick labels only add clutter.
    if HIDE_INNER_TICKLABELS:
        if col_idx > 0:
            ax_top.set_ylabel("")
            ax_top.tick_params(labelleft=False)
            ax_bot.set_ylabel("")
            ax_bot.tick_params(labelleft=False)

out = HERE / OUTPUT_PNG
fig.savefig(out, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out}")


# Solve-time comparison
fig_t, ax_t = plt.subplots(figsize=(8, 5), constrained_layout=True)
for i, (fn, title) in enumerate(TIME_FILES):
    it, times = load_iter_series(HERE / fn)
    if it is None or len(it) == 0:
        continue
    ax_t.plot(it, times, marker="o", ms=4, lw=1.6,
              color=cmap10(i % 10), label=title)

ax_t.set_xlabel("Time (h)")
ax_t.set_ylabel("Solve time (s)")
ax_t.legend(loc="best", fontsize=8, framealpha=0.9)
out_t = HERE / OUTPUT_TIME_PNG
fig_t.savefig(out_t, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_t}")


# Lyapunov comparison
lyapunov_runs = []
for fn, title in RUN_FILES:
    data = load_lyapunov_series(HERE / fn)
    if data is not None:
        lyapunov_runs.append((title, data))

if lyapunov_runs:
    v_ylim = Y_LIMITS_OVERRIDE.get("lyapunov", shared_limits([
        data[col].values
        for _, data in lyapunov_runs
        for col in ("V_finite", "V_infinite", "V_total")
    ]))
    fig_v, axes_v = plt.subplots(
        1,
        len(lyapunov_runs),
        figsize=(6.5 * len(lyapunov_runs), 5),
        constrained_layout=True,
        squeeze=False,
    )
    for idx, (ax, (title, data)) in enumerate(zip(axes_v[0], lyapunov_runs)):
        t = data["t [h]"]
        ax.plot(t, data["V_finite"], lw=1.8, label=r"$\mathrm{V}_{\mathrm{finite}}$")
        ax.plot(t, data["V_infinite"], lw=1.8, label=r"$\mathrm{V}_{\mathrm{infinite}}$")
        ax.plot(t, data["V_total"], lw=2.0, label=r"$\mathrm{V}_{\mathrm{total}}$")
        add_panel_label(ax, idx)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Lyapunov value")
        if v_ylim is not None:
            ax.set_ylim(*v_ylim)
        if HIDE_INNER_TICKLABELS and idx > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.legend(prop={"family": FONT_FAMILY})
    out_v = HERE / OUTPUT_LYAPUNOV_PNG
    fig_v.savefig(out_v, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_v}")


# Total-cost comparison
cost_runs = []
for fn, title in COST_RUNS:
    _, _, pwr = load_run(HERE / fn)
    pwr = pwr.iloc[:min(X_MAX_HOURS + 1, len(pwr))].reset_index(drop=True)
    cost_runs.append((title, pwr))

cost_labels = [title for title, _ in cost_runs]
total_costs = [
    compute_cost_kwh(pwr, pwr_cols, dt=3600, power_to_kw=POWER_SCALE)
    for _, pwr in cost_runs
]
fig_cost, ax_cost = plt.subplots(
    figsize=(max(8, 1.8 * len(cost_runs)), 5.5),
    constrained_layout=True,
)
bars = ax_cost.bar(
    np.arange(len(cost_runs)),
    total_costs,
    color=[cmap10(i % 10) for i in range(len(cost_runs))],
)
ax_cost.set_xticks(np.arange(len(cost_runs)), cost_labels)
ax_cost.set_ylabel("Energy (kWh)")
ax_cost.tick_params(axis="x", labelsize=8)
ax_cost.bar_label(
    bars, labels=[f"{cost:,.0f}" for cost in total_costs],
    padding=3, fontsize=9,
)
cost_min = min(total_costs)
cost_max = max(total_costs)
cost_span = cost_max - cost_min
cost_pad = 0.15 * (cost_span or 1.0)
ax_cost.set_ylim(cost_min - cost_pad, cost_max + cost_pad)
out_cost = HERE / OUTPUT_COST_PNG
fig_cost.savefig(out_cost, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_cost}")


def plot_horizon_cost_comparison(end_hour, out_file):
    horizon_costs = [
        compute_cost_through_hour(
            pwr, pwr_cols, end_hour=end_hour, dt=3600,
            power_to_kw=POWER_SCALE,
        )
        for _, pwr in cost_runs
    ]
    if any(cost is None for cost in horizon_costs):
        print(f"Skipped {out_file}: not enough rows through {end_hour} h")
        return

    fig_h, ax_h = plt.subplots(
        figsize=(max(8, 1.8 * len(cost_runs)), 5.5),
        constrained_layout=True,
    )
    bars_h = ax_h.bar(
        np.arange(len(cost_runs)),
        horizon_costs,
        color=[cmap10(i % 10) for i in range(len(cost_runs))],
    )
    ax_h.set_xticks(np.arange(len(cost_runs)), cost_labels)
    ax_h.set_ylabel("Energy (kWh)")
    ax_h.tick_params(axis="x", labelsize=8)
    ax_h.bar_label(
        bars_h, labels=[f"{cost:,.0f}" for cost in horizon_costs],
        padding=3, fontsize=9,
    )
    cost_min_h = min(horizon_costs)
    cost_max_h = max(horizon_costs)
    cost_span_h = cost_max_h - cost_min_h
    cost_pad_h = 0.15 * (cost_span_h or 1.0)
    ax_h.set_ylim(cost_min_h - cost_pad_h, cost_max_h + cost_pad_h)
    out_h = HERE / out_file
    fig_h.savefig(out_h, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_h}")


plot_horizon_cost_comparison(6, OUTPUT_COST_6H_PNG)
plot_horizon_cost_comparison(12, OUTPUT_COST_12H_PNG)


# Results summary
def write_results(path):
    lines = []
    for fn, title in TIME_FILES:
        _, _, pwr = load_run(HERE / fn)
        pwr = pwr.iloc[:min(X_MAX_HOURS + 1, len(pwr))].reset_index(drop=True)
        cpu_seconds = load_cpu_time_seconds(HERE / fn)
        cost = compute_cost_kwh(
            pwr, pwr_cols, dt=3600, power_to_kw=POWER_SCALE,
        )
        cost_12h = compute_cost_through_hour(
            pwr, pwr_cols, end_hour=12, dt=3600,
            power_to_kw=POWER_SCALE,
        )
        cost_60h = compute_cost_through_hour(
            pwr, pwr_cols, end_hour=60, dt=3600,
            power_to_kw=POWER_SCALE,
        )
        cost_12h_str = (
            f"{cost_12h:.2f} kWh" if cost_12h is not None else "N/A"
        )
        cost_60h_str = (
            f"{cost_60h:.2f} kWh" if cost_60h is not None else "N/A"
        )
        cpu_str = format_cpu_time(cpu_seconds) if cpu_seconds is not None else "N/A"
        mpc0, mpc_last = load_first_last_iter_seconds(HERE / fn)
        mpc0_str = format_cpu_time(mpc0) if mpc0 is not None else "N/A"
        mpc_last_str = format_cpu_time(mpc_last) if mpc_last is not None else "N/A"
        flat_title = title.replace("\n", " — ")
        lines.append(f"{flat_title} [{fn}]")
        lines.append(f"    Total cost: {cost:.2f}")
        lines.append(
            f"    0-12 h cost (including t=12): {cost_12h_str}")
        lines.append(
            f"    0-60 h cost (including t=60): {cost_60h_str}")
        lines.append(f"    CPU time:   {cpu_str}")
        lines.append(f"    mpc0:       {mpc0_str}")
        lines.append(f"    mpc_last:   {mpc_last_str}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path}")


write_results(HERE / OUTPUT_TXT)

plt.show()
