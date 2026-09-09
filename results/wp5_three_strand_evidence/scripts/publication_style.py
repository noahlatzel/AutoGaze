"""Outcome-neutral Matplotlib defaults for WP5."""
from pathlib import Path

PALETTE = {"black":"#000000","grey":"#7A7A7A","orange":"#E69F00","sky_blue":"#56B4E9","green":"#009E73","yellow":"#F0E442","blue":"#0072B2","vermillion":"#D55E00","purple":"#CC79A7"}

def apply_publication_style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":10,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","savefig.bbox":"tight"})

def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
