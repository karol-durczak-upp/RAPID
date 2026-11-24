import numpy as np

def audit(X,Z,c,rtol=1e-12):
    X,Z,c=map(lambda x:np.asarray(x,dtype=float),(X,Z,c))
    if (X.ndim != 2 or Z.ndim != 2 or c.ndim != 1 or
            X.shape != (Z.shape[0], c.size) or not all(X.shape)):
        raise ValueError("Incompatible or empty design dimensions")
    if not all(np.isfinite(a).all() for a in (X,Z,c)) or not np.any(c):
        raise ValueError("Inputs must be finite and contrast nonzero")
    if not 0 < rtol < 1:
        raise ValueError("rtol must lie between zero and one")
    u,s,v=np.linalg.svd(Z,full_matrices=False)
    k=np.count_nonzero(s>rtol*(s[0] if s.size else 0))
    Q=u[:,:k]
    A=X-Q@(Q.T@X)
    U,S,V=np.linalg.svd(A,full_matrices=False)
    keep=S>rtol*np.linalg.norm(X,2)
    R=V[keep].T
    d=c-R@(R.T@c)
    distance=np.linalg.norm(d)/np.linalg.norm(c)
    out=dict(rank=int(sum(keep)),distance=distance,basis=R,singular_values=S,rtol=rtol,estimable=distance<=rtol)
    if out['estimable']:
        w=U[:,keep]@((R.T@c)/S[keep])
        out.update(w=w,res=(np.max(np.abs(X.T@w-c)),np.max(np.abs(Z.T@w),initial=0)))
    else:
        h=d/(c@d)
        g=-v[:k].T@((Q.T@(X@h))/s[:k])
        out.update(h=h,g=g,res=(np.max(np.abs(X@h+Z@g)),abs(c@h-1)))
    return out

def block_update(basis, mean, new, c, rtol=1e-12):
    """Apply Eq. (5) to an existing block and return rank and target distance."""
    basis,mean,new,c = map(lambda a:np.asarray(a,dtype=float),(basis,mean,new,c))
    v=new-mean
    d=v-basis@(basis.T@v)
    if np.linalg.norm(d)>rtol*max(np.linalg.norm(mean),np.linalg.norm(new)):
        basis=np.column_stack((basis,d/np.linalg.norm(d)))
    return basis.shape[1],np.linalg.norm(c-basis@(basis.T@c))/np.linalg.norm(c)
