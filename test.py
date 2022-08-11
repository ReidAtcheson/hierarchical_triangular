#import python unit test framework
import unittest
from htri import *
import numpy as np
import pdb



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
        self.assertLess( res , 1e-11 )
    def test_hierarchical_triangular_trans(self):
        m=128
        minm=8
        k=1
        seed=23498723
        rng=np.random.default_rng(seed)
        htri=make_htri_rng_dense(0,m,minm,k,rng)
        L=htri_to_dense(htri,0,m,minm,k)
        out=eval_htri_trans(htri,0,m,minm,L.T)
        self.assertLess(jnp.linalg.norm(out - jnp.eye(m)) , 1e-11 )
    def test_hierarchical_id_matvec0(self):
        m=128
        minm=8
        k=1
        seed=23498723
        htri=make_htri_id(0,m,minm,k)
        I=jnp.eye(m)
        LI = matvec_htri(htri,0,m,minm,I)
        self.assertLess( jnp.linalg.norm(LI - I), 1e-11 )
    def test_hierarchical_id_matvec1(self):
        m=128
        minm=8
        k=1
        htri=make_htri_id(0,m,minm,k)
        seed=23498723
        rng=np.random.default_rng(seed)
        X=rng.uniform(-1,1,size=(m,3))
        LX = matvec_htri(htri,0,m,minm,X)
        self.assertLess( jnp.linalg.norm(LX - X), 1e-11 )


    def test_hierarchical_matvec(self):
        m=128
        minm=8
        k=1
        seed=23498723
        rng=np.random.default_rng(seed)
        htri=make_htri_rng_dense(0,m,minm,k,rng)
        X=rng.uniform(-1,1,size=(m,3))
        LX = matvec_htri(htri,0,m,minm,X)
        Xh = eval_htri(htri,0,m,minm,LX)        
        self.assertLess( jnp.linalg.norm(X - Xh), 1e-11 )





unittest.main()
