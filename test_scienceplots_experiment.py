"""Publication-plot and reuse checks; no additional registration experiments."""
import ast
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import numerical_experiments as core
import numerical_experiments_scienceplots as experiment
from research_plot_style import research_style


class SciencePlotsExperimentTests(unittest.TestCase):
    def test_only_config_dimensions_change_from_validated_defaults(self):
        old,new=asdict(core.ExperimentConfig()),asdict(experiment.CONFIG)
        self.assertEqual(new.pop('dimensions'),(2,4,8,16))
        old.pop('dimensions')
        self.assertEqual(new,old)
        self.assertIs(experiment.core.run_trial,core.run_trial)
        self.assertIs(experiment.core.aggregate_results,core.aggregate_results)

    def test_frame_metric_and_one_best_improvement_solve_are_reused(self):
        core.validate_frame_error()
        code=ast.parse(Path(core.__file__).read_text())
        functions={n.name:n for n in code.body if isinstance(n,ast.FunctionDef)}
        register=functions['register_point_clouds']
        constructors=[n for n in ast.walk(register) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='TrustRegions']
        self.assertEqual(len(constructors),1)
        self.assertNotIn('For',{type(n).__name__ for n in ast.walk(register)})
        greedy=functions['greedy_sign_refinement']
        # Acceptance occurs after the nested i,j scans, not inside either pair loop.
        move_loop=next(n for n in greedy.body if isinstance(n,ast.For))
        pair_scan=next(n for n in move_loop.body if isinstance(n,ast.For))
        self.assertFalse(any(isinstance(n,ast.Break) for n in ast.walk(pair_scan)))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='TrustRegions' for n in ast.walk(greedy)))

    def test_log_sd_crossing_zero_is_flagged_without_clipping_statistics(self):
        mean=np.array([1.,2.,3.]);std=np.array([.5,2.,4.])
        lo,hi,shown=experiment.errorbar_plan(mean,std,True)
        np.testing.assert_array_equal(lo,[.5,0.,-1.])
        np.testing.assert_array_equal(hi,[1.5,4.,7.])
        np.testing.assert_array_equal(shown,[True,False,False])
        np.testing.assert_array_equal(experiment.errorbar_plan(mean,std,False)[2],[True,True,True])
        np.testing.assert_array_equal(mean,[1.,2.,3.]);np.testing.assert_array_equal(std,[.5,2.,4.])
        with self.assertRaisesRegex(ValueError,'nonpositive mean'):
            experiment.errorbar_plan([0.],[.1],True)

    def test_actual_scienceplots_style_and_no_latex(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import scienceplots
        self.assertIn('science',plt.style.available)
        self.assertIn('no-latex',plt.style.available)
        old=dict(plt.rcParams)
        with research_style():
            self.assertFalse(plt.rcParams['text.usetex'])
            self.assertEqual(plt.rcParams['axes.linewidth'],.5)
            self.assertEqual(plt.rcParams['pdf.fonttype'],42)
        self.assertEqual(dict(plt.rcParams),old)

    def test_sample_sd_and_separate_distribution_dimension_groups(self):
        rows=[]
        for kind in experiment.CONFIG.dataset_types:
            for dimension in experiment.CONFIG.dimensions:
                for trial in range(20):
                    rows.append({'dataset_type':kind,'dimension':dimension,
                                 **{key:float(trial+1) for key in core.METRICS}})
        summaries=core.aggregate_results(rows)
        self.assertEqual(len(summaries),12)
        for row in summaries:
            self.assertEqual(row['num_trials'],20)
            self.assertEqual(row['frame_error_mean'],10.5)
            self.assertEqual(row['frame_error_std'],float(np.arange(1,21).std(ddof=1)))

    def test_all_figures_match_aggregate_numeric_positions_colors_and_log_bounds(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        summaries=[]
        for kind in experiment.CONFIG.dataset_types:
            for dimension in experiment.CONFIG.dimensions:
                summaries.append({'dataset_type':kind,'dimension':dimension,'num_trials':20,
                                  **{metric+'_mean':.1 for metric in core.METRICS},
                                  **{metric+'_std':.2 if kind=='gaussian_ellipsoid' and dimension==8 else .01 for metric in core.METRICS}})
        colors={}
        seen=[]
        def inspect(fig,path,**kwargs):
            ax=fig.axes[0]
            seen.append(str(path))
            for line in ax.lines:
                if line.get_label() not in core.DATASET_LABELS.values():continue
                np.testing.assert_array_equal(line.get_xdata(),[2,4,8,16])
                np.testing.assert_array_equal(line.get_ydata(),[.1]*4)
                if line.get_label() in colors:self.assertEqual(line.get_color(),colors[line.get_label()])
                colors[line.get_label()]=line.get_color()
            if ax.get_yscale()=='log':
                for line in ax.lines:self.assertTrue(np.all(np.asarray(line.get_ydata())>0))
                for collection in ax.collections:
                    if hasattr(collection,'get_segments'):
                        for segment in collection.get_segments():self.assertTrue(np.all(segment[:,1]>0))
        with patch.object(Figure,'savefig',inspect),patch.object(experiment,'write_csv') as write:
            records=experiment.plot_results(summaries,Path('unused_mock_output'))
        self.assertEqual(len(seen),12)
        self.assertEqual(len(records),72)
        self.assertEqual(sum(r['axis_scale']=='log' and not r['lower_whisker_drawn'] for r in records),2)
        self.assertEqual(len(colors),3)
        write.assert_called_once()


if __name__=='__main__':
    unittest.main()
