# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import time as tm
import sys
import matplotlib.pyplot as plt

N = 200


# Classical Matrix_Product

def Matrix_Prod(M,N):
    if M.shape[1] == N.shape[0]:
        K = M.shape[1]
        Prod = np.zeros((M.shape[0], N.shape[1]))
        for i in range(M.shape[0]):
            for j in range(N.shape[1]):
                for k in range(K):
                    Prod[i,j] += M[i,k] * N[k,j]
        return Prod
    else:
        print("Sorry! The product is not possible")


 


# Product by block

def Bloc_Product(A, B, block_size):
    # Test of product possibility
    if A.shape[1] == B.shape[0]:

    # Initialization of C
        C = np.zeros((A.shape[0], B.shape[1]))
        n, m = A.shape
        m, p = B.shape
    

    # Do the product  bloc by bloc
        for i in range(0, n, block_size):
            for j in range(0, p, block_size):
                for k in range(0, m, block_size):
                    # Product of current bloc
                    C[i:i+block_size, j:j+block_size] += np.dot(A[i:i+block_size, k:k+block_size], B[k:k+block_size, j:j+block_size])
                
        return C
    else:
        print("Sorry")
        
M = np.random.randint(1,9,[N,N])
L = np.random.randint(1,7,[N,N])



tim = []
l = []
for i in range(2,20,2):
    
    l.append(i)
    start = tm.time()
    
    Bloc_Product(M,L,i)
    
    end = tm.time()
    
    msec = (end - start)*1000
    
    print(msec)
    
    rate = sys.getsizeof(float) * N * N * (1000.0 / msec) / (1024*1024)
    
    tim.append(rate)
    

for i in range(len(tim)):
    plt.plot(l[i],tim[i],'*')
plt.plot(l,tim)


