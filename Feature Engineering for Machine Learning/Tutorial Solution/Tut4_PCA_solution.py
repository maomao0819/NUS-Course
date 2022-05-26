import numpy as np
import pandas as pd

data = pd.read_csv('seeds_with_headers.csv')

print(data.head())

print(data.describe())
print(data.columns)

meanData = np.mean(data, axis=0)
print(meanData)

covMat = np.cov(data - meanData, rowvar=False)
print('covMat')
print(covMat)

print('covMatPandas = data.cov()')
covMatPandas = data.cov()
print(covMatPandas.cov())

eVal, eVect = np.linalg.eig(covMat)

eVal_sorted_index = np.argsort(eVal)[::-1]
eVal_sorted = eVal[eVal_sorted_index]

print('eVal')
print(eVal)
print('eVal_sorted_index')
print(eVal_sorted_index)
print('eVal_sorted')
print(eVal_sorted)
print('eVect')
print(eVect)

nCol, nRow = eVect.shape
print(eVect.shape)

for i in range(nCol):
    print(i)
    print(eVect[:, i])

dotProd1 = np.dot(eVect, eVect.T)
print(dotProd1)

dotProd2 = np.dot(eVect.T, eVect)
print(dotProd2)

# --- project data onto principal components

coef = np.zeros([data.shape[0], eVect.shape[1]])

for i in range(data.shape[0]):
    d = data.iloc[i, :]
    coef[i, :] = np.dot(d, eVect)

print(coef)

# --- from the coefficients, reconstruct the data
recData = np.zeros(data.shape)

for i in range(recData.shape[0]):
    recData[i, :] = np.dot(eVect, coef[i, :].T)

print('data')
print(data)
print('recData')
print(recData)

# --- calculate the reconstruction errors
recErrors = np.zeros([data.shape[0], 1])

for i in range(data.shape[0]):
    temp = (data.iloc[i, :] - recData[i, :])
    recErrors[i] = np.dot(temp, temp.T)

print('reconstruction errors =================================')
print(recErrors)

# --- now, repeat the above exercise by using just 4 of the eigenvectors corresponding
# to the biggest eigenvalues

# --- from the coefficients, reconstruct the data
recData = np.zeros(data.shape)

for i in range(recData.shape[0]):
    recData[i, :] = np.dot(eVect[:, 0:3], coef[i, 0:3].T)

print('data')
print(data)
print('recData with 4 eigenvectors corresponding to biggest eigenvalues')
print(recData)

# --- calculate the reconstruction errors
recErrors = np.zeros([data.shape[0], 1])

for i in range(data.shape[0]):
    temp = (data.iloc[i, :] - recData[i, :])
    recErrors[i] = np.dot(temp, temp.T)

print('-------------------------------------------------------')
print('reconstruction errors with 4 eigenvectors corresponding to biggest eigenvalues')
print(recErrors)

# --- now, repeat again the above exercise by using just 4 of the eigenvectors corresponding
# to the smallest eigenvalues

# --- from the coefficients, reconstruct the data
recData = np.zeros(data.shape)

for i in range(recData.shape[0]):
    recData[i, :] = np.dot(eVect[:, -4:], coef[i, -4:].T)

print('data')
print(data)
print('recData with 4 eigenvectors corresponding to biggest eigenvalues')
print(recData)

# --- calculate the reconstruction errors
recErrors = np.zeros([data.shape[0], 1])

for i in range(data.shape[0]):
    temp = (data.iloc[i, :] - recData[i, :])
    recErrors[i] = np.dot(temp, temp.T)

print('=================================================================')
print('reconstruction errors with 4 eigenvectors corresponding to smallest eigenvalues')
print(recErrors)



