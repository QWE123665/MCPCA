import numpy as np 
import mcpca
from mcpca import similarity_measures


p = 5
k = 3
r = 4

A = np.random.randn(5,4)
A = A/np.linalg.norm(A,axis=0)
B = np.abs(np.random.randn(3,4))
X_list = []
Nlist = [100,150,80]
for i in range(len(Nlist)):
    Sigma = A @ np.diag(B[i,:]) @ A.T 
    X = np.random.multivariate_normal(np.zeros(A.shape[0]), Sigma, size=Nlist[i])
    X_list.append(X)

model = mcpca.MCPCA(n_components = None, rank_range = [1,2,3,4])
model.fit(X_list)
Z_list = model.transform(X_list, return_list=True)
print("Ascore:")
print(similarity_measures(model.components_,A)[0])
