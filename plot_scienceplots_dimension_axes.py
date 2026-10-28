"""Plot-only axis comparison from the completed 2/4/8/16 aggregate CSV.

No experiment/core imports, data generation, aggregation or optimization.
All previous files are hashed and preserved; every new filename must be unused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys

import numpy as np

from research_plot_style import research_style

ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = ROOT / "numerical_results/scienceplots_dimensions_2_4_8_16"
DIMENSIONS = (2, 4, 8, 16)
LABELS = {"gaussian_ellipsoid": "Gaussian ellipsoid", "uniform_box": "Uniform box", "laplace_product": "Laplace product"}
FORMATS = (("o", "-"), ("s", "--"), ("^", "-."))
METRICS = {"frame_error": "Raw frame error (Frobenius norm)",
           "normalized_frame_error": r"Normalized frame error ($E_{\mathrm{frame}}/\sqrt{d}$)"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stem(metric, variant):
    return f"{metric.replace('_', '-')}-vs-dimension-scienceplots-{variant}-base2"


def load_aggregates(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r["dataset_type"], int(r["dimension"])) for r in rows]
    expected = {(kind, d) for kind in LABELS for d in DIMENSIONS}
    if len(keys) != 12 or set(keys) != expected or any(int(r["num_trials"]) != 20 for r in rows):
        raise ValueError("Expected exactly 12 saved distribution/dimension groups, 20 trials each.")
    for row in rows:
        for metric in METRICS:
            mean, sd = float(row[metric+"_mean"]), float(row[metric+"_std"])
            if not np.isfinite([mean, sd]).all() or mean <= 0 or sd < 0:
                raise ValueError("Cannot plot these saved mean/SD values without changing statistics.")
    return rows


def run(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, LogLocator, LogFormatterSciNotation

    source = directory / "aggregated_results.csv"
    rows = load_aggregates(source)
    destinations = [directory / (stem(metric, variant)+suffix) for metric in METRICS
                    for variant in ("loglog", "semilogx") for suffix in (".png", ".pdf")]
    destinations += [directory/name for name in ("dimension_axes_plot_data.csv", "dimension_axes_plot_validation.json", "PLOTTING_AXES_COMPARISON.md")]
    if any(path.exists() for path in destinations):
        raise ValueError("A new plot/audit filename already exists; refusing to overwrite.")
    original = {str(path.resolve()): digest(path) for path in directory.iterdir() if path.is_file()}
    plot_rows, figure_checks = [], []
    with research_style():
        assert not plt.rcParams["text.usetex"]
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for metric, ylabel in METRICS.items():
            for variant in ("loglog", "semilogx"):
                log_y = variant == "loglog"
                fig, ax = plt.subplots(figsize=(6.8, 4.25), layout="constrained")
                omissions = []
                ax.set_xscale("log", base=2)
                ax.set_yscale("log", base=10) if log_y else ax.set_yscale("linear")
                for (kind, label), (marker, linestyle), color in zip(LABELS.items(), FORMATS, palette):
                    series = sorted((r for r in rows if r["dataset_type"] == kind), key=lambda r:int(r["dimension"]))
                    x = np.array([int(r["dimension"]) for r in series])
                    mean = np.array([float(r[metric+"_mean"]) for r in series])
                    sd = np.array([float(r[metric+"_std"]) for r in series])
                    lower, upper = mean-sd, mean+sd
                    drawable = lower > 0 if log_y else np.ones(len(x), dtype=bool)
                    curve, = ax.plot(x, mean, color=color, marker=marker, linestyle=linestyle,
                                     markersize=5, linewidth=1.3, label=label)
                    np.testing.assert_array_equal(curve.get_xdata(), x)
                    np.testing.assert_array_equal(curve.get_ydata(), mean)
                    if drawable.any():
                        bars = ax.errorbar(x[drawable], mean[drawable], yerr=sd[drawable],
                                           fmt="none", capsize=3, color=color, elinewidth=.9)
                        segments = bars.lines[2][0].get_segments()
                        for segment, i in zip(segments, np.flatnonzero(drawable)):
                            np.testing.assert_array_equal(segment, [[x[i], lower[i]], [x[i], upper[i]]])
                    for i in np.flatnonzero(~drawable):
                        # A negative/zero lower endpoint cannot be represented on
                        # log y. Omit it explicitly rather than choosing a floor.
                        bars = ax.vlines(x[i], mean[i], upper[i], color=color, linewidth=.9)
                        np.testing.assert_array_equal(bars.get_segments()[0], [[x[i], mean[i]], [x[i], upper[i]]])
                        ax.plot(x[i], upper[i], marker="_", color=color, markersize=6)
                        ax.plot(x[i], mean[i], marker="v", markerfacecolor="none", markeredgecolor=color,
                                markersize=9, linestyle="none")
                        omissions.append({"dataset_type":kind,"dimension":int(x[i])})
                    for i in range(len(x)):
                        plot_rows.append({"metric":metric,"variant":variant,"x_scale":"log2","y_scale":"log10" if log_y else "linear",
                                          "dataset_type":kind,"dimension":int(x[i]),"num_trials":20,
                                          "mean":float(mean[i]),"sample_std":float(sd[i]),
                                          "mean_minus_sd":float(lower[i]),"mean_plus_sd":float(upper[i]),
                                          "lower_whisker_drawn":bool(drawable[i]),"color":color,"marker":marker,"line_style":linestyle})
                ax.xaxis.set_major_locator(FixedLocator(DIMENSIONS))
                ax.xaxis.set_major_formatter(FixedFormatter([str(d) for d in DIMENSIONS]))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_xlim(2 / 2**.12, 16 * 2**.12)
                ax.set(xlabel=r"Dimension $d$ (log$_2$ scale)",
                       ylabel=ylabel+(" (log scale)" if log_y else ""))
                if log_y:
                    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
                    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
                else:
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3,4), useMathText=True)
                ax.set_title(r"Mean $\pm$ 1 sample SD; 20 trials per distribution/dimension", fontsize=10)
                ax.grid(which="major", alpha=.22, linewidth=.5)
                handles, labels = ax.get_legend_handles_labels()
                if omissions:
                    handles.append(Line2D([],[],marker="v",markerfacecolor="none",markeredgecolor="black",linestyle="none"))
                    labels.append(r"Mean $-$ SD $\leq0$: lower bar omitted")
                ax.legend(handles, labels, loc="best", fontsize=9, frameon=False)
                fig.canvas.draw()
                pixels = ax.transData.transform(np.column_stack((DIMENSIONS, np.ones(4))))[:,0]
                np.testing.assert_allclose(np.diff(pixels), np.repeat(np.diff(pixels)[0],3), rtol=0, atol=1e-9)
                assert [tick.get_text() for tick in ax.get_xticklabels()] == ["2","4","8","16"]
                assert ax.xaxis.get_transform().base == 2
                if log_y:
                    for line in ax.lines:
                        assert np.all(np.asarray(line.get_ydata()) > 0)
                    for collection in ax.collections:
                        if hasattr(collection, "get_segments"):
                            assert all(np.all(segment[:,1] > 0) for segment in collection.get_segments())
                png, pdf = [directory/(stem(metric,variant)+suffix) for suffix in (".png", ".pdf")]
                fig.savefig(png,dpi=600); fig.savefig(pdf)
                pdf_data = pdf.read_bytes()
                assert pdf_data.startswith(b"%PDF-") and b"%%EOF" in pdf_data[-100:]
                assert b"/FontFile2" in pdf_data and b"/Subtype /Image" not in pdf_data
                assert len(re.findall(rb"/Type /Page\s",pdf_data)) == 1
                figure_checks.append({"metric":metric,"variant":variant,"png":str(png),"pdf":str(pdf),
                                      "x_axis_base":2,"y_axis_base":10 if log_y else None,
                                      "dimension_labels":["2","4","8","16"],"equal_x_tick_pixel_spacing":True,
                                      "means_and_errorbar_endpoints_match_aggregates":True,
                                      "omitted_nonpositive_lower_whiskers":omissions,
                                      "pdf_vector_with_embedded_truetype":True})
                plt.close(fig)
    # An extra round-trip check confirms every audit value equals the saved CSV.
    lookup = {(r["dataset_type"], int(r["dimension"])):r for r in rows}
    for plotted in plot_rows:
        saved = lookup[plotted["dataset_type"],plotted["dimension"]]
        for field, suffix in (("mean","_mean"),("sample_std","_std")):
            assert plotted[field] == float(saved[plotted["metric"]+suffix])
    with (directory/"dimension_axes_plot_data.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(plot_rows[0]));writer.writeheader();writer.writerows(plot_rows)
    changed = [path for path,value in original.items() if digest(path)!=value]
    assert not changed, changed
    assert not any(name in sys.modules for name in ("numerical_experiments","pymanopt"))
    validation = {"passed":True,"source_csv":str(source),"source_csv_sha256":digest(source),
                  "styles":["science","no-latex"],"SciencePlots_version":metadata.version("SciencePlots"),
                  "png_dpi":600,"plotted_records":len(plot_rows),"figures":figure_checks,
                  "previous_file_hashes":original,"previous_files_unchanged":True,
                  "changed_previous_files":changed,"data_generation_or_optimization_run":False,
                  "numerical_core_and_pymanopt_not_imported":True}
    (directory/"dimension_axes_plot_validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    notes = """# Plot-only logarithmic x-axis comparison

Input: unchanged aggregated_results.csv from the completed 240-trial experiment.
No data generation, aggregation or optimization. Only saved arithmetic means
and sample SDs (ddof=1) are drawn. Both raw and normalized frame-error plots
are provided as log-log (base-2 x, base-10 y) and semilog-x (base-2 x, linear y).
All ticks are explicitly 2, 4, 8, 16, equally spaced in log-x display coordinates.

The original *-scienceplots-log.png/pdf semilog-y figures remain unchanged.
New files end in -scienceplots-loglog-base2.png/pdf or
-scienceplots-semilogx-base2.png/pdf. Styles: SciencePlots ['science','no-latex'];
600-dpi PNG and vector PDF with embedded TrueType fonts. Colors, markers and
line styles consistently identify each distribution.

On log y, Gaussian d=8 has mean-SD<=0 for both raw and normalized metrics.
Its lower SD whisker is omitted and flagged by a hollow down-triangle plus
legend entry; the mean and full upper SD whisker are retained. No clipping,
positive floor or change to CSV statistics is used. Semilog-x shows the full
arithmetic SD interval, including its negative lower bound (a statistical
interval bound, not a negative observed error).

Log-log is the preferred main view: doubling steps have equal spacing and
vertical spacing emphasizes multiplicative changes across dimensions.
Semilog-x preserves equal doubling steps but emphasizes absolute differences
and variability; Gaussian d=8's large SD dominates the linear y range, making
the small d=2 errors harder to separate. Normalized log-log also clearly shows
the Gaussian mean declining from d=8 to d=16 while the other means increase.
None of these axis changes provides a new statistical significance test.

dimension_axes_plot_data.csv records all plotted means, SDs and lower-bound
flags. dimension_axes_plot_validation.json records exact source-value checks,
equal tick-spacing checks, vector-PDF checks and hashes confirming every
pre-existing file in this experiment directory remains unchanged.
No Overleaf files or previous figures/results were edited.
"""
    (directory/"PLOTTING_AXES_COMPARISON.md").write_text(notes,encoding="utf-8")
    print(f"Saved four new figures (PNG + vector PDF). All {len(plot_rows)} plotted means/SDs match the CSV; {len(original)} previous files unchanged.")
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=DEFAULT_DIRECTORY)
    run(parser.parse_args().directory.resolve())


if __name__=="__main__":
    main()
