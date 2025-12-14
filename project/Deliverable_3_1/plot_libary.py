import matplotlib.pyplot as plt
import scienceplots
from cycler import cycler
import numpy as np
import matplotlib as mpl

# 1) Apply SciencePlots first (it may set usetex)
plt.style.use(["science", "nature"])

# 2) NOW override anything that could trigger TeX
mpl.rcParams.update({
    "text.usetex": False,          # <- must be after style.use
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],

    # Internal math rendering (no external LaTeX)
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",

    # Bold title support
    "axes.titleweight": "bold",

    # color-blind safe palette
    "axes.prop_cycle": cycler(color=[
        "#0052cc", # blue: trajectories
        "#D55E00", #red-orange: sampled points / highlight
        "#009E73", #greenish: set X region
        "#e6ab02", #yellow: O_inf region
        "#000000", #black: axes, text
    ]),

    # Clean ticks / lines
    "lines.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Example data
x = np.linspace(0, 10, 200)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot
fig, ax = plt.subplots(figsize=(3.4, 2.5))

ax.plot(x, y1, label=r"$\sin(x)$")
ax.plot(x, y2, label=r"$\cos(x)$")

# Bold title
ax.set_title("Bold Title Example", fontweight='bold')

# Sans-serif labels
ax.set_xlabel(r"$v_s$ [m s$^{-1}$]")
ax.set_ylabel(r"$\alpha_{\mathrm{eff}}$")

ax.legend(loc='best', frameon=False)
#ax.tick_params(direction="out", length=4, width=0.8)

plt.tight_layout()
plt.show()
