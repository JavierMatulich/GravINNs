"""Figure helpers shared by the trainers.

Origin: ``_save_panels_separately`` in notebook cell 1, plus a small wrapper
around ``IPython.display`` so the trainer also runs from a plain script.
"""
import os


def save_panels_separately(fig, axes, names, out_dir, prefix, dpi=300):
    """Save each axis in `axes` as its own high-resolution PNG, cropped to
    that panel's tight bounding box (title, legend, ticks included)."""
    os.makedirs(out_dir, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    paths = []
    for ax, name in zip(axes, names):
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches=bbox)
        paths.append(path)
    return paths


_save_panels_separately = save_panels_separately   # notebook name


class _NullDisplay:
    """Stand-in for an IPython DisplayHandle when there is no notebook."""
    def update(self, *_args, **_kw):
        pass


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        # Jupyter, JupyterLab, VS Code and Colab all run an IPython *kernel*
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False


def live_display(fig, live_plot=None):
    """Return an object with ``.update(fig)``.

    live_plot=None  -> live figure in Jupyter, silent elsewhere (default)
    live_plot=True  -> always use IPython.display (the notebook behaviour)
    live_plot=False -> never display (panels are still saved to disk)
    """
    if live_plot is None:
        live_plot = in_notebook()
    if not live_plot:
        return _NullDisplay()
    from IPython.display import display
    return display(fig, display_id=True)
