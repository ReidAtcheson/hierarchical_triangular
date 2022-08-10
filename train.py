import optax
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from htri import *
import pickle


def make_banded_matrix(m,diag,bands,rng):
    subdiags=[rng.uniform(-1,1,m) for _ in bands] + [rng.uniform(0.1,1,m) + diag] + [rng.uniform(-1,1,m) for _ in bands]
    offs = [-x for x in bands] + [0] + [x for x in bands]
    return sp.diags(subdiags,offs,shape=(m,m))




seed=2234432
rng=np.random.default_rng(seed)

#Set up sparse matrix
m=256
diag=4.0
A=make_banded_matrix(m,diag,[1,2,3,10,40,100],rng)
A = 0.5*(A+A.T)

#Set up initial hierarchical triangular matrix
minm = 32
k=1
htri = make_htri_id(0,m,minm,k)
d = jnp.ones((m,1))
params = (d,htri)


#Set up gradient descent
batchsize=8
lr=1e-3
opt=optax.sgd(lr)
opt_state = opt.init(params)
nepochs=1000

#Keep track of best params so far
best_params=params
best_err=1e20

#Optional eigenvalue printing
print_eigs=True


errs=[]
for it in range(nepochs):
    start=time.time()
    x=np.array(rng.uniform(size=(m,batchsize)))
    Ax=A@x
    x=jnp.array(x)
    Ax=jnp.array(Ax)
    #Compute update
    g = grad(loss)(params,minm,Ax,x)
    updates,opt_state = opt.update(g,opt_state)
    params = optax.apply_updates(params,updates)
    stop=time.time()
    #Compute new error
    err = loss(params,minm,Ax,x)
    print(f"it = {it},     elapsed = {stop-start : .4f},    loss = {err : 4f}")
    if errs and err<best_err:
        best_params=params
        best_err=err
    errs.append(err)

    if print_eigs and it%10==0:
        Afull=A.toarray()
        MA = eval_inv(params,minm,Afull)
        itstr = str(it).zfill(5)
        plt.close()
        eigMA = la.eigvals(MA)
        plt.scatter(np.real(eigMA),np.imag(eigMA))
        plt.xlim([-6,6])
        plt.title("Eigenvalues of preconditioned operator")
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.savefig(f"eigs/{itstr}.png")



plt.semilogy(errs)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.savefig("loss.svg")
plt.close()


#Output training results
f=open("params.dat","wb")
pickle.dump(best_params,f)
f=open("A.dat","wb")
pickle.dump(A,f)

