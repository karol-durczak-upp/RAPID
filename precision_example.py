"""Reproduce the precision extension using the unchanged RAPID 0.1.0 audit API.
Run with --source /path/to/RAPID-0.1.0/src, or install rapid_audit first.
The assumed error covariance is sigma**2 times the identity.
"""
import argparse, sys, json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--out',default='precision_results.json');a=p.parse_args()
if a.source:sys.path.insert(0,a.source)
from rapid_audit import audit
c=np.array([1.,-1.,0.]); Z=np.ones((2,1)); rows=[]
for eps in (1.,.01,.001):
 X=np.array([[0.,0.,0.],[eps,-eps,0.]])
 r=audit(X,Z,c); w=r['w']; expected=np.array([-1.,1.])/eps
 assert r['estimable'] and r['rank']==1
 np.testing.assert_allclose(w,expected,rtol=1e-11,atol=1e-11)
 # Independent least-squares solution of original certificate constraints.
 direct=np.linalg.lstsq(np.vstack((X.T,Z.T)),np.r_[c,0.],rcond=None)[0]
 np.testing.assert_allclose(w,direct,rtol=1e-11,atol=1e-11)
 gain=float(np.linalg.norm(w)); variance=float(w@w)
 np.testing.assert_allclose(variance,2/eps**2,rtol=1e-11)
 assert max(r['res'])<1e-9
 rows.append(dict(epsilon=eps,rank=r['rank'],estimable=bool(r['estimable']),distance=float(r['distance']),weights=w.tolist(),noise_gain=gain,variance_factor=variance,sigma=.01,contrast_standard_deviation=.01*gain,certificate_residuals=list(map(float,r['res']))))
# Monte Carlo check on a separate fixed seed for each design.
for i,row in enumerate(rows):
 noise=np.random.default_rng(20261119+i).normal(0,.01,(200000,2))
 empirical=float(np.std(noise@np.array(row['weights']),ddof=1))
 row['monte_carlo_sd']=empirical
 assert abs(empirical/row['contrast_standard_deviation']-1)<.01
out=dict(note='Unchanged RAPID 0.1.0 API; analytical and original-constraint checks; 200000 Gaussian draws per design.',numpy=np.__version__,results=rows)
with open(a.out,'w') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
