#import python unit test framework
import unittest
from htri import *
import numpy as np



class TestHierarchicalTriangular(unittest.TestCase):
    def test_hierarchical_triangular(self):
        m=128
        minm=8
        k=1
        seed=23498723
        rng=np.random.default_rng(seed)
        htri=make_htri_rng_dense(0,m,minm,k,rng)
        L=htri_to_dense(htri,0,m,minm,k)
        out=eval_htri(htri,0,m,minm,L)
        res = jnp.linalg.norm(out - jnp.eye(m)) 
        self.assertTrue( res < 1e-11 )
    def test_hierarchical_triangular_trans(self):
        m=128
        minm=8
        k=1
        seed=23498723
        rng=np.random.default_rng(seed)
        htri=make_htri_rng_dense(0,m,minm,k,rng)
        L=htri_to_dense(htri,0,m,minm,k)
        out=eval_htri_trans(htri,0,m,minm,L.T)
        self.assertTrue( jnp.linalg.norm(out - jnp.eye(m)) < 1e-11 )




unittest.main()
