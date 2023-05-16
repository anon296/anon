import numpy as np


n_factors = (6,6,2,3,3,40,40)

y = np.arange(6)

def combinations_by_num(nfs):
    y = np.arange(nfs[0])
    for nf in nfs[1:]:
        new = np.arange(nf)
        y = np.c_[np.repeat(y,nf,0),np.tile(new,len(y))]
    assert y.shape[1] == len(nfs)
    assert y.shape[0] == np.prod(nfs)
    return y

def combinations_recursive(arrs):
    if len(arrs)==1:
        return np.expand_dims(arrs[0],1)
    head = arrs[0]
    tail = combinations(arrs[1:])
    return np.c_[np.repeat(head,len(tail),0), np.tile(tail,(len(head),1))]

def combinations(arrs):
    y = arrs[0]
    for new in arrs[1:]:
        y = np.c_[np.repeat(y,len(new),0),np.tile(new,len(y))]
    return y

#arrs = [np.arange(nf) for nf in n_factors]
#assert (combinations_by_num(n_factors) == combinations(arrs)).all()

print('making labels...')
y = combinations_by_num(n_factors)
import pdb; pdb.set_trace()  # XXX BREAKPOINT
nids=(5,6,2,0,1,3,4)
y = y[:,nids]

print('loading images...')
X = np.load('datasets/mpi3d_real.npz')['images']
#X = X.reshape(*n_factors,64,64,3)

N = np.prod(n_factors)

print('saving datasets...')
small_mask = np.random.choice(N,size=1000,replace=False)
np.savez('datasets/mpi3d_small.npz',imgs=X[small_mask],latents_classes=y[small_mask])

med_mask = np.random.choice(N,size=10000,replace=False)
np.savez('datasets/mpi3d_medium.npz',imgs=X[small_mask],latents_classes=y[med_mask])

np.savez('datasets/mpi3d.npz',imgs=X,latents_classes=y)
