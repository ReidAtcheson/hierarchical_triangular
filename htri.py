import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax.experimental import sparse
from jax import random
import jax.nn as jnn
import jax.scipy.linalg as jla
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from functools import partial
import optax
from jax.config import config
config.update("jax_enable_x64", True)
#@partial(jit, static_argnums=[0,2])


#Create hierarchical block triangular matrix equivalent to identity
def make_htri_id(bm,em,minm,k):
    size=em-bm
    if size<minm:
        return jnp.zeros((size,size))
    else:
        s2=size//2
        #(left,lr block, right)
        nrows=em-(bm+s2)
        ncols=s2
        U=jnp.zeros((nrows,k))
        Vt=jnp.zeros((k,ncols))
        return (make_htri_id(bm,bm+s2,minm,k),(U,Vt),make_htri_id(bm+s2,em,minm,k))

#Create hierarchical block triangular matrix with random entries 
def make_htri_rng_dense(bm,em,minm,k,rng):
    size=em-bm
    if size<minm:
        D=np.tril(rng.uniform(1,2,size=(size,size)),k=-1) + jnp.eye(size)
        return jnp.array(D)
    else:
        s2=size//2
        #(left,lr block, right)
        nrows=em-(bm+s2)
        ncols=s2
        U=jnp.array(rng.uniform(-1,1,size=(nrows,k)))
        Vt=jnp.array(rng.uniform(-1,1,size=(k,ncols)))
        lbm=bm
        lem=bm+s2
        rbm=bm+s2
        rem=em

        return (make_htri_rng_dense(bm,bm+s2,minm,k,rng),(U,Vt),make_htri_rng_dense(bm+s2,em,minm,k,rng))

#Convert a hierarchical block triangular matrix into a dense triangular matrix
def htri_to_dense(htri,bm,em,minm,k):
    def htri_to_dense_rec(htri,bm,em,minm,k,L):
        m,_ = L.shape
        size=em-bm
        if size<minm:
            return L.at[bm:em,bm:em].set(htri)
        else:
            s2=size//2
            left,lr,right = htri
            lbm=bm
            lem=bm+s2
            rbm=bm+s2
            rem=em
            #Eliminate top block
            L = htri_to_dense_rec(left,lbm,lem,minm,k,L)
            U,Vt=lr
            L = L.at[rbm:rem,lbm:lem].set(U@Vt)
            L = htri_to_dense_rec(right,rbm,rem,minm,k,L)
            return L
    m=em-bm
    L=jnp.zeros((m,m))
    return htri_to_dense_rec(htri,bm,em,minm,k,L)





def eval_htri(htri,bm,em,minm,x):
    def eval_htri_rec(htri,bm,em,minm,x):
        m,ncols=x.shape
        size=em-bm
        if size<minm:
            return x.at[bm:em,:].set(jla.solve_triangular(htri,x[bm:em,:],lower=True,unit_diagonal=True))
        else:
            s2=size//2
            left,lr,right = htri
            lbm=bm
            lem=bm+s2
            rbm=bm+s2
            rem=em
            #Eliminate top block
            x = eval_htri_rec(left,lbm,lem,minm,x)
            U,Vt=lr
            #Eliminate low rank factor
            x = x.at[rbm:rem,:].add(-U@(Vt@(x[lbm:lem,:])))
            #Eliminate bottom block
            x = eval_htri_rec(right,rbm,rem,minm,x)
            return x
    out=x.copy()
    out = eval_htri_rec(htri,bm,em,minm,out)
    return out




m=128
minm=8
k=1
seed=23498723
rng=np.random.default_rng(seed)
#htri = make_htri_id(0,m,32,1)
htri=make_htri_rng_dense(0,m,minm,k,rng)
L=htri_to_dense(htri,0,m,minm,k)
out=eval_htri(htri,0,m,minm,L)
print(jnp.linalg.norm(out-jnp.eye(m)))
