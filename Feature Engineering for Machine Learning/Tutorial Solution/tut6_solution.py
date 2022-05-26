import numpy as np

nodes = 12

A = [[0 for j in range(nodes)] for i in range(nodes)]
A = np.array(A)
#---------[A  B  C  D  E  F  G  H  I  J  K  L ------
A[0,:]  = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # A
A[1,:]  = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # B
A[2,:]  = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # C
A[3,:]  = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # D
A[4,:]  = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # E
A[5,:]  = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # F
A[6,:]  = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # G
A[7,:]  = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]  # H
A[8,:]  = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]  # I
A[9,:]  = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # J
A[10,:] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # K
A[11,:] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # L


print('------ A -----------')
for i in range(nodes):
    print(A[i])

D = [[0 for j in range(nodes)] for i in range(nodes)]

for i in range(nodes):
    D[i][i] = sum(A[i])

print('------ D -----------')
for i in range(nodes):
    print(D[i])

D = np.array(D)

L = D - A

print('------ L -----------')
for i in range(nodes):
    print(L[i])

#---- compute Lx, L*L*x, L*L*L*x
x = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
x = np.array(x).T
print(x)

Lx = L @ x
print('Lx ------')
print(Lx)

L2x = L @ L @ x
print('L*L*x ------')
print(L2x)

L3x = L @ L @ L @ x
print('L*L*L*x -----')
print(L3x)