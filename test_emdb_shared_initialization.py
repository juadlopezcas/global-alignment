"""Controlled initialization geometry/protocol tests before full EMDB fitting."""
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import emdb_shared_initialization_comparison as experiment


class SharedInitializationTests(unittest.TestCase):
    def setUp(self):
        self.X=np.arange(72,dtype=float).reshape(24,3)/72
        self.Y=self.X[::-1].copy()
        self.R=Rotation.from_rotvec([.4,-.2,.7]).as_matrix()

    def test_icp_receives_exact_supplied_rotation_and_zero_translation(self):
        def fake(X,Y,**kwargs):
            self.assertIs(X,self.X);self.assertIs(Y,self.Y)
            np.testing.assert_array_equal(kwargs['initial_transform'][:3,:3],self.R)
            np.testing.assert_array_equal(kwargs['initial_transform'][:3,3],np.zeros(3))
            self.assertEqual(kwargs['max_correspondence_distance'],2.)
            return experiment.core.MethodResult(self.R,.1,{'icp_initial_transformation':json.dumps(kwargs['initial_transform'].tolist())})
        with patch.object(experiment.core,'run_icp',side_effect=fake) as run:
            experiment.icp_refinement(self.X,self.Y,self.R)
        run.assert_called_once()

    def test_cpd_pre_rotation_and_noncommuting_composition_preserve_scale_translation(self):
        increment=Rotation.from_rotvec([-.3,.5,.1]).as_matrix()
        translation=np.array([.02,-.01,.03]);scale=.8
        def fake(X,Y):
            np.testing.assert_array_equal(X,self.X@self.R.T)
            self.assertIs(Y,self.Y)
            return experiment.core.MethodResult(increment,.1,{'cpd_scale':scale,'cpd_translation':json.dumps(translation.tolist())})
        with patch.object(experiment.core,'run_rigid_cpd',side_effect=fake) as run:
            result=experiment.cpd_refinement(self.X,self.Y,self.R)
        run.assert_called_once()
        np.testing.assert_allclose(result.R_est,increment@self.R,atol=1e-15)
        self.assertGreater(np.linalg.norm(result.R_est-self.R@increment),.01)
        predicted=scale*(self.X@result.R_est.T)+translation
        two_stage=scale*((self.X@self.R.T)@increment.T)+translation
        np.testing.assert_allclose(predicted,two_stage,atol=1e-15)
        np.testing.assert_array_equal(result.diagnostics['cpd_effective_initial_R'],self.R)

    def test_initialization_is_only_two_pca_calls_and_original_best_greedy(self):
        Xframe=Rotation.from_rotvec([.1,.2,.3]).as_matrix()
        Yframe=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()
        signs=np.array([-1,-1,1])
        pca=lambda F:experiment.core.PCAResult(F,4,1e-9,'tolerance')
        greedy=experiment.core.GreedyResult(signs,[-.01,-.02],[(0,1)],.01,6,True)
        with patch.object(experiment.core,'estimate_pca_frame',side_effect=[pca(Xframe),pca(Yframe)]) as pca_fit, \
             patch.object(experiment.core,'greedy_sign_refinement',return_value=greedy) as greedy_fit, \
             patch.object(experiment.core,'TrustRegions',side_effect=AssertionError('Initialization must not run final optimization')):
            R,diagnostics=experiment.shared_initialization(self.X,self.Y,2330588202912440548)
        self.assertEqual(pca_fit.call_count,2);greedy_fit.assert_called_once()
        self.assertIs(pca_fit.call_args_list[0].args[0],self.X)
        self.assertIs(pca_fit.call_args_list[1].args[0],self.Y)
        self.assertIs(pca_fit.call_args_list[0].args[1],pca_fit.call_args_list[1].args[1])
        np.testing.assert_array_equal(R,(Yframe*signs)@Xframe.T)
        experiment.core.assert_rotation(R)
        self.assertFalse(R.flags.writeable)
        self.assertEqual(diagnostics['greedy_num_pair_evaluations'],6)

    def test_fit_interfaces_never_receive_true_rotation(self):
        for function in (experiment.shared_initialization,experiment.mmd_refinement,experiment.icp_refinement,experiment.cpd_refinement):
            parameters=inspect.signature(function).parameters
            self.assertNotIn('R_true',parameters);self.assertNotIn('truth',parameters)

    def test_known_rotation_synthetic_validation(self):
        checks=experiment.synthetic_validation()
        self.assertTrue(checks['passed'])
        self.assertTrue(checks['cpd_public_R_rejects_valid_SO3'])
        for result in checks['results'].values():self.assertLess(result['rotation_error'],1e-5)

    def test_full_ours_split_pipeline_matches_frozen_reference_without_duplicate_init(self):
        pairs=experiment.dataset.read_manifest(experiment.frozen.DEFAULT_MANIFEST)
        pair=pairs[0]
        X,Y,truth,scale=experiment.frozen.prepare_pair(pair,experiment.frozen.DEFAULT_MANIFEST)
        with patch.object(experiment.core,'run_goicp',side_effect=AssertionError('Never rerun Go-ICP')):
            R,initialization=experiment.shared_initialization(X,Y,int(pair['ours_fit_seed']))
            result=experiment.mmd_refinement(X,Y,R)
        old=next(r for r in experiment.dataset.read_manifest(experiment.REFERENCE) if r['pair_id']==pair['pair_id'] and r['method']=='ours')
        self.assertLess(experiment.verify_mmd_reproduction(result,old),1e-9)
        self.assertAlmostEqual(experiment.core.rotation_angle_error_degrees(truth,result.R_est),float(old['rotation_angle_error_degrees']),places=7)
        self.assertEqual(result.diagnostics['final_TrustRegions_solves'],1)


if __name__=='__main__':
    unittest.main()
