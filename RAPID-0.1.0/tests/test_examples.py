import unittest
import numpy as np
from examples.run import extrusion, synthetic
from rapid_audit import audit, block_update

class Examples(unittest.TestCase):
    def test_extrusion(self):
        camp, intercept = extrusion()
        self.assertEqual([camp["rank"], intercept["rank"]], [0, 1])
        self.assertFalse(camp["estimable"] or intercept["estimable"])
        np.testing.assert_allclose(camp["g"], [-.5, -.04], atol=1e-12)
        self.assertAlmostEqual(intercept["distance"], .962008694)
        for result in (camp, intercept):
            self.assertLess(max(result["res"]), 1e-10)

    def test_synthetic(self):
        for (result, (rank, dist)), (r, e) in zip(synthetic(), [(0,False),(1,False),(1,True)]):
            self.assertEqual((result["rank"], result["estimable"]), (r,e))
            self.assertEqual(rank,r)
            self.assertAlmostEqual(dist,result["distance"])
            self.assertLess(max(result["res"]),1e-12)

    def test_existing_nonzero_block_space(self):
        X=np.array([[0,0,0],[1,0,0],[9,9,9]],float)
        Z=np.array([[1,0],[1,0],[0,1]],float)
        c=np.array([0,1,0],float)
        base=audit(X,Z,c)
        for v, expected in [([2,0,0],(1,False)),([2,0,1],(2,False)),([2,1,0],(2,True))]:
            full=audit(np.vstack([X,v]),np.vstack([Z,[1,0]]),c)
            rank,dist=block_update(base["basis"],X[:2].mean(axis=0),v,c)
            self.assertEqual((rank,dist < 1e-12),expected)
            self.assertEqual((full["rank"],full["estimable"]),expected)
            self.assertAlmostEqual(full["distance"],dist)

    def test_redundant_nuisance_columns(self):
        result=audit([[0],[1]],[[1,1],[1,1]],[1])
        self.assertTrue(result["estimable"])
        self.assertLess(max(result["res"]),1e-12)
