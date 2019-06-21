import numpy as np
from sklearn.covariance import graphical_lasso, empirical_covariance

l1_norm = lambda matrix_B: np.abs(matrix_B).sum()

def comp_matrix_empirical_covariance(N, X, Y, B):
    return  1 / N *(Y-X.dot(B)).T.dot(Y-X.dot(B))

def comp_matrix_omega(emp_cov, lambda1):
    matrix_omega = graphical_lasso(emp_cov, lambda1, mode='lars')
    return matrix_omega

def iterator_omega(N, X, Y, param_B, lambda1):
    return comp_matrix_omega(comp_matrix_empirical_covariance(N, X, Y, param_B), lambda1)[1]

def ridge_matrix_B(X, Y, lambda2):
    P = X.shape[1]
    B_ridge = np.linalg.inv(X.T.dot(X) + lambda2 * np.diag([1.0]* P)).dot(X.T).dot(Y)
    return B_ridge

def comp_matrix_B(N, X, Y, B, Omega, lambda2, epsilon=1e-6, max_iter_num=1e5):
    S = X.T.dot(X)
    H = X.T.dot(Y).dot(Omega)
    
    size1, size2 = B.shape
    B_ridge = ridge_matrix_B(X, Y, lambda2)
    
    ridge_l1_norm = l1_norm(B_ridge)
    
    iter_num = 1
    while True:
        iter_num += 1
        
        B_old = B
        B = B.copy()

        for r in range(size1):
            for c in range(size2):
                U = S.dot(B).dot(Omega)
                res1 = B[r,c] + (H[r,c]- U[r,c] + 1e-6) / (S[r,r] * Omega[c,c] + 1e-6)
                res2 = abs(res1)-(0.5 * N * lambda2 + 1e-6) / (S[r,r] * Omega[c,c] + 1e-6)
                U[r, c] = U[r, c] - B[r, c] * S[r, r] * Omega[c, c]
                # for the sake of avoiding numeric error "python int too large to C long"
                B[r, c] = np.sign(res1) * max(0, res2)
                U[r, c] += B[r, c] * S[r, r] * Omega[c, c]
                
#         print(l1_norm(B), ridge_l1_norm)
        if  l1_norm(B - B_old) < epsilon * ridge_l1_norm or iter_num >= max_iter_num:
            break
        
    return B

def training(lambda1, lambda2, X, Y, epsilon=0.0001):
    P = X.shape[1]
    N = X.shape[0]
    Q = Y.shape[1]
    
    B_ridge = ridge_matrix_B(X, Y, lambda2)
    ridge_l1_norm = l1_norm(B_ridge)
    
    matrix_B_param = np.array([[0.0 for i in range(Q)] 
                                 for j in range(P)]) * 0.0
    
    while True:
        matrix_B_old = matrix_B_param
        matrix_Omega = iterator_omega(N, X, Y, matrix_B_param, lambda1)
        matrix_B_param = comp_matrix_B(N, X, Y, matrix_B_param, matrix_Omega, 
                                       lambda2, 0.0001)
        
#         print(l1_norm(matrix_B_param - matrix_B_old))
        if l1_norm(matrix_B_param - matrix_B_old) < epsilon * ridge_l1_norm:
            break
            
    return matrix_B_param, B_ridge

# matrix_Omega = iterator_omega(N, matrix_X, matrix_Y, matrix_B, 0.3)