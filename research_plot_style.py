"""Default style for NEW research figures; never restyle old outputs implicitly."""
from contextlib import contextmanager


@contextmanager
def research_style():
    import matplotlib.pyplot as plt
    import scienceplots  # Registers the real package's styles (required).

    with plt.style.context(["science", "no-latex"]):
        # Embed TrueType fonts in vector exports; no external TeX dependency.
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            yield
