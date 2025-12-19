import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import scienceplots

def set_mpc_style():
    # Apply SciencePlots first
    plt.style.use(["science", "nature"])

    # Override TeX and fonts
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],

        "mathtext.fontset": "dejavusans",
        "mathtext.default": "it",

        "axes.titleweight": "bold",

        "axes.prop_cycle": cycler(color=[
            "#003a7d",  # blue
            "#ff9d3a",  # orange
            "#d83034",  # red
            "#008dff",  # mid-blue
            "#4ecb8d",  # green
            "#c701ff",  # pruple
        ]),

        "lines.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })
