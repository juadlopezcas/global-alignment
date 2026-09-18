"""Generator and grouping checks; existing frame-error tests remain untouched."""

import unittest

import numpy as np

import numerical_experiments as experiments


class DatasetGeneratorTests(unittest.TestCase):
    def test_gaussian_refactor_preserves_exact_legacy_draws(self):
        legacy_rng = np.random.default_rng(913)
        n, d, noise_std = 60, 5, 0.05
        expected_X = legacy_rng.standard_normal((n, d)) * np.sqrt(np.arange(d, 0, -1))
        expected_R = experiments.random_rotation(d, legacy_rng)
        expected_Y = expected_X @ expected_R.T + noise_std * legacy_rng.standard_normal((n, d))
        expected = (expected_X, expected_Y, expected_R)
        actual = experiments.generate_dataset(n, d, "gaussian_ellipsoid", np.random.default_rng(913), noise_std)
        wrapper = experiments.generate_ellipsoid(n, d, np.random.default_rng(913), noise_std)
        for want, got, compatible in zip(expected, actual, wrapper):
            np.testing.assert_array_equal(want, got)
            np.testing.assert_array_equal(want, compatible)

    def test_all_generators_preserve_row_rotation_convention(self):
        for dataset_type in experiments.DATASET_TYPES:
            with self.subTest(dataset_type=dataset_type):
                X, Y, rotation = experiments.generate_dataset(80, 3, dataset_type, np.random.default_rng(17), 0.0)
                self.assertEqual(X.shape, (80, 3))
                self.assertTrue(np.isfinite(X).all())
                experiments.assert_rotation(rotation)
                np.testing.assert_allclose(Y, X @ rotation.T, atol=1e-12)

    def test_source_moments_and_distinct_marginal_laws(self):
        diagnostics = experiments.validate_generators()
        self.assertEqual({row["dataset_type"] for row in diagnostics}, set(experiments.DATASET_TYPES))
        kurtoses = {row["dataset_type"]: row["mean_marginal_kurtosis"] for row in diagnostics}
        self.assertLess(kurtoses["uniform_box"], kurtoses["gaussian_ellipsoid"] - 0.5)
        self.assertLess(kurtoses["gaussian_ellipsoid"], kurtoses["laplace_product"] - 1.0)

    def test_rngs_are_order_independent_and_gaussian_compatible(self):
        def collect(names, dimensions, trials):
            results = {}
            for name in names:
                for dimension in dimensions:
                    for trial in trials:
                        data_rng, fit_rng = experiments.trial_rngs(42, dimension, trial, name)
                        results[name, dimension, trial] = (data_rng.standard_normal(20), fit_rng.standard_normal(20))
            return results

        first = collect(experiments.DATASET_TYPES, (2, 5), (0, 1))
        reordered = collect(reversed(experiments.DATASET_TYPES), (5, 2), (1, 0))
        for key in first:
            for original, repeated in zip(first[key], reordered[key]):
                np.testing.assert_array_equal(original, repeated)
        self.assertEqual(len({streams[0].tobytes() for streams in first.values()}), len(first))
        self.assertEqual(len({stream.tobytes() for streams in first.values() for stream in streams}), 2 * len(first))
        legacy_seeds = np.random.SeedSequence([42, 5, 1]).spawn(2)
        for legacy_seed, actual in zip(legacy_seeds, first["gaussian_ellipsoid", 5, 1]):
            np.testing.assert_array_equal(np.random.default_rng(legacy_seed).standard_normal(20), actual)

    def test_dataset_configuration_rejects_unsupported_and_duplicate_names(self):
        for names in ((), ("normal",), ("uniform_box", "uniform_box")):
            with self.subTest(names=names), self.assertRaises(ValueError):
                experiments.ExperimentConfig(dataset_types=names).validate()
        with self.assertRaises(ValueError):
            experiments.generate_dataset(20, 2, "normal", np.random.default_rng(8))
        config = experiments.ExperimentConfig()
        self.assertEqual(config.dataset_types, experiments.DATASET_TYPES)
        self.assertEqual(len(config.dataset_types) * len(config.dimensions) * config.num_trials, 300)


class DatasetAggregationTests(unittest.TestCase):
    def test_datasets_and_dimensions_are_not_pooled(self):
        rows = []
        for dataset_type, offset in (("gaussian_ellipsoid", 0), ("uniform_box", 10), ("laplace_product", 20)):
            for dimension in (2, 5):
                for delta in (0, 2):
                    rows.append({"dataset_type": dataset_type, "dimension": dimension,
                                 **{metric: offset + dimension + delta for metric in experiments.METRICS}})
        summaries = experiments.aggregate_results(rows)
        self.assertEqual(len(summaries), 6)
        for row in summaries:
            offset = {"gaussian_ellipsoid": 0, "uniform_box": 10, "laplace_product": 20}[row["dataset_type"]]
            self.assertEqual(row["num_trials"], 2)
            for metric in experiments.METRICS:
                self.assertEqual(row[metric + "_mean"], offset + row["dimension"] + 1)
                self.assertAlmostEqual(row[metric + "_std"], np.sqrt(2))

    def test_single_trial_sd_remains_undefined(self):
        row = {"dataset_type": "uniform_box", "dimension": 2,
               **{metric: 1.0 for metric in experiments.METRICS}}
        summary = experiments.aggregate_results([row])[0]
        self.assertTrue(all(np.isnan(summary[metric + "_std"]) for metric in experiments.METRICS))


if __name__ == "__main__":
    unittest.main()
