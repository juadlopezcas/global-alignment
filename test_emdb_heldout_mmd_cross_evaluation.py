"""Evaluation-only numerical, seed, interval and checkpoint checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import t

import diagnose_emdb_heldout_mmd_cross_evaluation as e


class HeldoutChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X, self.Y = rng.normal(size=(37,3))*.2, rng.normal(size=(41,3))*.2
        self.R = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        self.rotations = {k: (self.R.copy() if k=='R_mmd' else np.eye(3)) for k in e.CANDIDATES}

    def test_blocked_dense_and_diagonal_exclusion(self):
        for chunk in (1,17,64):
            _,_,losses,deltas=e.evaluate_pair(self.X,self.Y,self.rotations,chunk)
            dense=e.core.RotationMMD(self.X,self.Y,1.)
            for k,R in self.rotations.items():
                self.assertAlmostEqual(losses[k],dense.evaluate(R),places=14)
                self.assertAlmostEqual(losses[k],e.core.mmd_squared(self.X@R.T,self.Y,1.),places=14)
            for k in e.CONTRASTS:
                self.assertAlmostEqual(deltas[k],dense.evaluate(self.rotations['R_'+k])-dense.evaluate(self.R),places=14)

    def test_negative_estimates_preserved_identical_rotations_zero(self):
        rr={k:np.eye(3) for k in e.CANDIDATES}
        _,_,losses,deltas=e.evaluate_pair(self.X,self.X,rr,17)
        self.assertTrue(all(v<0 for v in losses.values()))
        self.assertTrue(all(v==0 for v in deltas.values()))

    def test_seed_streams_unique_and_reproducible(self):
        pairs=e.sampling.read_manifest(e.frozen.DEFAULT_MANIFEST)
        a,b=e.sampling_plan(pairs),e.sampling_plan(pairs)
        self.assertEqual(a,b);self.assertEqual(len(a),700)
        seeds=[r[k] for r in a for k in ('source_sampling_seed','target_sampling_seed')]
        self.assertEqual(len(set(seeds)),1400)
        self.assertEqual(sum(r['num_points']==1000 for r in a),400)
        self.assertEqual(sum(r['num_points']==5000 for r in a),200)
        self.assertEqual(sum(r['num_points']==10000 for r in a),100)

    def test_reuses_sampling_protocol_and_shared_radius(self):
        pop=np.arange(600,dtype=float).reshape(200,3)/13
        plan={'num_points':37,'source_sampling_seed':1234,'target_sampling_seed':5678}
        X,Y,d=e.previous.sample_pair(pop,plan,self.R)
        X2,Y2,d2=e.previous.sample_pair(pop,plan,self.R)
        np.testing.assert_array_equal(X,X2);np.testing.assert_array_equal(Y,Y2)
        self.assertEqual(d,d2)
        self.assertFalse(X.flags.writeable);self.assertFalse(Y.flags.writeable)
        np.testing.assert_allclose(X.mean(axis=0),0,atol=1e-15)
        self.assertAlmostEqual(max(np.linalg.norm(X,axis=1).max(),np.linalg.norm(Y,axis=1).max()),1.)

    def test_paired_student_t_and_training_reversal(self):
        train={'pair_000':{'L_'+k:1e-5 for k in e.CONTRASTS}}
        train['pair_000']['L_mmd']=0.
        ids=[{'pair_id':'pair_000','candidate':k,'angular_error_degrees':0.} for k in e.CANDIDATES]
        values=np.array([-1.,-2.,-3.])*1e-6
        rows=[{'pair_id':'pair_000','num_points':1000,**{'Delta_'+k:v for k in e.CONTRASTS}} for v in values]
        summaries=e.summarize(rows,train,ids)
        for row in summaries:
            half=t.ppf(.975,2)*values.std(ddof=1)/np.sqrt(3)
            self.assertAlmostEqual(row['ci95_mean_lower'],values.mean()-half,places=17)
            self.assertTrue(row['training_preference_reversed'])
            self.assertEqual(row['alternative_preferred_repetitions'],3)
            self.assertFalse(row['complete'])

    def test_atomic_checkpoint_resume_rejects_tampering(self):
        entry={'evaluation_id':'pair_000_n1000_rep000','pair_id':'pair_000','num_points':1000}
        row={**entry,'candidate_set_sha256':'fixed','sigma':1.,**{'L_'+k[2:]:0. for k in e.CANDIDATES},
             **{'Delta_'+k:0. for k in e.CONTRASTS}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.csv'
            e.checkpoint_rows(p,[row])
            self.assertEqual(len(e.load_rows(p,[entry],'fixed')),1)
            with self.assertRaises(ValueError):e.load_rows(p,[entry],'wrong')
            text=p.read_text().replace('pair_000_n1000','pair_001_n1000')
            p.write_text(text)
            with self.assertRaises(ValueError):e.load_rows(p,[entry],'fixed')

    def test_fitting_and_rotation_cleanup_are_forbidden(self):
        for module,name in ((e.core,'run_icp'),(e.shared,'mmd_refinement'),(e.core,'clean_goicp_rotation'),(e.sampling,'random_rotation')):
            with self.assertRaises(AssertionError):
                with e.evaluation_only_guard():getattr(module,name)()


if __name__=='__main__':unittest.main()
