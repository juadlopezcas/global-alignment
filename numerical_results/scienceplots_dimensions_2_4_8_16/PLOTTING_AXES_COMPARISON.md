# Plot-only logarithmic x-axis comparison

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
