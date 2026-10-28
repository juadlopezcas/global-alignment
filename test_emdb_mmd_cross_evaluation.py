"""Evaluation-only scientific checks; never fit or sample new data."""
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_emdb_mmd_cross_evaluation as experiment


class MMDCrossEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with experiment.evaluation_only_guard():
            cls.pairs,cls.rows,cls.initializations,cls.prepared,_=experiment.load_validated_inputs()

    def test_fixed_numerical_tolerance_classification(self):
        tol=experiment.LOSS_TOLERANCE
        self.assertEqual(experiment.classify_alternative(2*tol),'MMD_lower')
        self.assertEqual(experiment.classify_alternative(-2*tol),'alternative_lower')
        for value in (-tol,0.,tol):self.assertEqual(experiment.classify_alternative(value),'numerical_tie')

    def test_guard_blocks_registration_and_sampling(self):
        for module,name in ((experiment.core,'run_icp'),(experiment.shared,'mmd_refinement'),(experiment.dataset,'sample_cloud')):
            with self.subTest(name=name), self.assertRaisesRegex(AssertionError,'forbidden'):
                with experiment.evaluation_only_guard():getattr(module,name)(None,None,None)

    def candidates(self,index=0):
        pair=self.pairs[index]
        initial=next(r for r in self.initializations if r['pair_id']==pair['pair_id'])
        return experiment.saved_candidates(pair,initial,self.rows,self.prepared[pair['pair_id']])

    def test_shared_not_identity_candidates_and_CPD_composition(self):
        rotations,lookup,residual=self.candidates()
        self.assertLess(residual,1e-12)
        np.testing.assert_array_equal(rotations['R_icp'],json.loads(lookup['icp_shared']['R_est']))
        self.assertGreater(np.linalg.norm(rotations['R_icp']-json.loads(lookup['icp_identity']['R_est'])),.1)
        d=json.loads(lookup['cpd_shared']['diagnostics'])
        np.testing.assert_allclose(rotations['R_cpd'],np.array(d['cpd_R_incremental'])@rotations['R_init'],atol=1e-12)

    def test_shared_arrays_rotation_only_loss_and_independent_full_MMD(self):
        rotations,lookup,_=self.candidates()
        X,Y,truth,_=self.prepared[self.pairs[0]['pair_id']]
        before={name:R.tobytes() for name,R in rotations.items()}
        original=experiment.core.RotationMMD.evaluate
        receipts=[]
        def checked(objective,R):
            self.assertIs(objective.X,X);self.assertIs(objective.Y,Y)
            receipts.append(R)
            return original(objective,R)
        with experiment.evaluation_only_guard(), \
             patch.object(experiment.core.RotationMMD,'evaluate',checked), \
             patch.object(experiment.core,'center_and_normalize',side_effect=AssertionError('No extra preprocessing')):
            values=experiment.evaluate_candidates(X,Y,rotations,truth)
        self.assertEqual(len(receipts),10)
        self.assertEqual(len(values),5)
        self.assertTrue(all(r['empirical_mmd']<0 for r in values))
        self.assertLess(max(r['independent_mmd_absolute_discrepancy'] for r in values),experiment.CONSISTENCY_TOLERANCE)
        self.assertEqual(before,{name:R.tobytes() for name,R in rotations.items()})
        mmd=next(r for r in values if r['candidate']=='R_mmd')
        self.assertEqual(mmd['empirical_mmd'],float(lookup['mmd_shared']['final_mmd']))
        self.assertNotIn('translation',inspect.signature(experiment.evaluate_candidates).parameters)
        self.assertNotIn('scale',inspect.signature(experiment.evaluate_candidates).parameters)

    def test_candidate_condition_mismatch_is_rejected(self):
        pair=self.pairs[0]
        initial=self.initializations[0]
        changed=[dict(r) for r in self.rows]
        next(r for r in changed if r['pair_id']==pair['pair_id'] and r['method_condition']=='icp_shared')['condition']='identity'
        with self.assertRaisesRegex(ValueError,'shared-start condition'):
            experiment.saved_candidates(pair,initial,changed,self.prepared[pair['pair_id']])

    def test_preference_counts_preserve_raw_differences_and_ties(self):
        tol=experiment.LOSS_TOLERANCE
        values=[{'candidate':name,'empirical_mmd':loss,'angular_error_degrees':angle} for name,loss,angle in
                [('R_init',1.,8.),('R_mmd',0.,6.),('R_icp',tol/2,1.),('R_cpd',2*tol,2.),('R_true',1e-5,0.)]]
        row=experiment.pair_differences('pair_test',values)
        self.assertEqual(row['Delta_icp'],tol/2)
        self.assertEqual(row['classification_icp'],'numerical_tie')
        self.assertFalse(row['MMD_lower_loss_but_higher_angle_than_icp'])
        self.assertTrue(row['MMD_lower_loss_but_higher_angle_than_cpd'])
        counts=experiment.lowest_counts([row])
        self.assertEqual(counts[0]['strict_lowest_among_three_refinements'],0)
        self.assertEqual(counts[0]['within_tolerance_of_lowest_among_three'],1)
        self.assertEqual(counts[1]['within_tolerance_of_lowest_among_three'],1)

    def test_pair017_saved_matrices_and_previous_loss(self):
        rotations,_,_=self.candidates(17)
        prior=json.loads((experiment.ROOT/'numerical_results/emdb_9515_mmd_diagnosis_pair_017/rotations.json').read_text())
        for name,key in (('R_init','R_greedy_initial'),('R_mmd','R_final'),('R_true','R_true')):
            np.testing.assert_array_equal(rotations[name],prior[key])
        X,Y,truth,_=self.prepared['pair_017']
        with experiment.evaluation_only_guard():values=experiment.evaluate_candidates(X,Y,rotations,truth)
        loss=next(r['empirical_mmd'] for r in values if r['candidate']=='R_mmd')
        self.assertEqual(loss,-0.0003940107467064191)


if __name__=='__main__':
    unittest.main()
