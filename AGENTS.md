# Research plotting preference

For all new research figures in this repository, including future shared-
initialization ICP/CPD comparisons, use the actual SciencePlots package with
`["science", "no-latex"]` by default. The shared `research_plot_style.research_style`
context provides this style without an external LaTeX dependency.

Preserve existing experiment scripts, CSVs and figures. Save new experiments
and figures separately; export publication figures as high-resolution PNG and
vector PDF. Document any nonpositive uncertainty bounds on logarithmic axes
instead of clipping or changing the saved statistics.
