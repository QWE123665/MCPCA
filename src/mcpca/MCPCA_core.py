import numpy as np
from .SPM21 import spm_21sym
from scipy.optimize import nnls
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm



# input a list of datasets with shared features, return the tensor of stacked covariance matrices
def cov_tensor(df_list):
    covs = []
    for df in df_list:
        M = np.cov(df,rowvar=False)
        covs.append(M)
    T = np.stack(covs,axis=2)
    return T


def similarity_measures(B,A):
    J=A.shape[1]
    columnlist=[]
    for i in range(J):
        v=A[:,i]
        dislist=[]
        signlist=[]
        for j in range(B.shape[1]):
            u=abs(v-B[:,j])
            uprime=abs(v+B[:,j])
            sign=1
            if np.sum(u**2)<np.sum(uprime**2):
                dislist.append(np.sum(u))
                signlist.append(sign)
            else:
                sign=-1
                dislist.append(np.sum(uprime**2))
                signlist.append(sign)
        a=min(dislist)
        index=dislist.index(a)
        sign=signlist[index]
        columnlist.append(sign*B[:,index])
        B=np.delete(B,index,1)
    permutedB=np.transpose(np.vstack(tuple(columnlist)))
    cosine_similarity=np.mean(np.sum(permutedB*A,axis=0))
    # for now the bound is 0.95
    return cosine_similarity,np.sort(np.sum(permutedB*A,axis=0))[::-1]


# input a tensor, return the choice of rank from a range of given ranks 
def choose_rank(T_tensor, rank_range, n_seed_pairs=5, cos_sim_threshold = 0.9,stats=False, scree_plot = False):
    if scree_plot:
        M = T_tensor.reshape(T_tensor.shape[0], -1)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        plt.figure()
        plt.plot(range(1, len(S) + 1), S, marker='o')
        plt.title('Scree Plot of Singular Values')
        plt.xlabel('Component')
        plt.ylabel('Singular Value')
        plt.grid()
        plt.show()
    rank_similarities = {}
    avg_similarities = []

    
    for r in rank_range:

        similarities_for_rank = []
        
        # Test multiple random seed pairs
        for pair_idx in range(n_seed_pairs):
            # Generate different random seed pairs for each test
            seed1 = pair_idx * 2 
            seed2 = pair_idx * 2 + 1
            
            # First decomposition
            np.random.seed(seed1)
            A_1, B_1 = spm_21sym(T_tensor, r)
            
            # Second decomposition  
            np.random.seed(seed2)
            A_2, B_2 = spm_21sym(T_tensor, r)
            
            # Compute cosine similarity between A matrices
            cosine_simA, sim_list = similarity_measures(A_1, A_2)
            similarities_for_rank.append(cosine_simA)
            
        
        # Calculate statistics for this rank
        avg_sim = np.mean(similarities_for_rank)
        std_sim = np.std(similarities_for_rank)
        min_sim = np.min(similarities_for_rank)
        max_sim = np.max(similarities_for_rank)
        
        rank_similarities[r] = {
            'similarities': similarities_for_rank,
            'mean': avg_sim,
            'std': std_sim,
            'min': min_sim,
            'max': max_sim
        }
        avg_similarities.append(avg_sim)
    avg_similarities = np.array(avg_similarities)
    if stats:
        return rank_range[np.where(avg_similarities>cos_sim_threshold)[0].max()],rank_similarities
    else:
        return rank_range[np.where(avg_similarities>cos_sim_threshold)[0].max()]


# input stack of covariance matrices T and rank r
# output MCPCs (matrix A) and loadings (matrix B)

def MCPCA_decompose(T,r, plot_A = False, plot_B = False):
    m, _, n = T.shape
    A, _ = spm_21sym(T,r)
    # Reshape tensor for matrix operations
    T_mat = T.reshape(m, -1)  # m × (m*n)
    
    # Compute Khatri-Rao product A ⊙ A
    A_kr = (A.reshape((-1,1,r)) * A.reshape((1,-1,r))).reshape(m**2,r)
    
    B_nnls = np.zeros((n, r))
    # Solve for each context (row of B) separately
    for i in range(n):
        # Extract the i-th slice: T[:,:,i]
        T_slice = T[:, :, i].flatten()  # Flatten to vector of length m²
        
        # Solve non-negative least squares: min ||A_kr * b_i - T_slice||²
        # subject to b_i >= 0
        b_i, residual = nnls(A_kr, T_slice)
        B_nnls[i, :] = b_i

    if plot_A:
        norm = TwoSlopeNorm(vcenter=0)
        plt.figure()
        plt.imshow(A, cmap='seismic', norm=norm)
        xlabels = [rf'$a_{{{i+1}}}$' for i in range(A.shape[1])]
        plt.xticks(ticks=np.arange(A.shape[1]), labels=xlabels, fontsize=8)
        plt.title("Heatmap of MCPC matrix A")
        plt.show()
        

    if plot_B:
        norm = TwoSlopeNorm(vcenter=0)
        plt.figure()
        plt.title("Heatmap of context loading matrix B")
        plt.imshow(B_nnls, cmap='seismic', norm=norm)
        xlabels = [rf'$b_{{{i+1}}}$' for i in range(B_nnls.shape[1])]
        ylabels = [rf'$X_{{{i+1}}}$' for i in range(B_nnls.shape[0])]
        plt.xticks(ticks=np.arange(B_nnls.shape[1]), labels=xlabels, fontsize=8)
        plt.yticks(ticks=np.arange(B_nnls.shape[0]), labels=ylabels, fontsize=12)
        plt.show()


    return A, B_nnls


if __name__ == '__main__':
    ## TEST 
    M1 = np.diag([1,2])
    M2 = np.diag([2,1])
    T = np.stack( [M1,M2],axis = 2)
    r = choose_rank(T,[1,2])
    A,B = MCPCA_decompose(T,r)
    print(A)
    print(B)
