# %%
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
# %%
dim = 20
nc = Dataset('test/resources/IASI-test-single-event.nc', 'r')
# take first average kernel
# avk = nc['state_WVatm_avk'][0, 0, 0]
avk_event = nc['state_WVatm_avk'][0]
avk = np.block(
    [[avk_event[0, 0], avk_event[1, 0]], 
    [avk_event[0, 1], avk_event[1, 1]]]
)




# %% [markdown]
# ### Eigendecomposotion
# Eigendecomposition of a Matrix $A$ has the form $A = Q Λ Q^{-1}$.

# #### Preconditions
# 1. $A$ should be square and diagonalizable

# %%
# eigenvalues are not necessarily ordered!
eigenvalues, eigenvectors = np.linalg.eig(avk.data)

# make shure imaginary part for first eigenvalues is near null
np.testing.assert_allclose(np.imag(eigenvalues[:dim]), 0)


# %%
# reduce dimension
Q = np.copy(np.real(eigenvectors[:, :dim]))
s = np.copy(np.real(eigenvalues[:dim]))
print(f'Q shape: {Q.shape}')
print(f's shape: {s.shape}')

# reconstruct old dimension
# Q.resize((28, 28))
# s.resize(28)
# print(f'Q shape: {Q.shape}')
# print(f's shape: {s.shape}')

Q_inv = np.linalg.pinv(Q)
s = np.diag(s)
print(f'Q_ shape: {Q_inv.shape}')
avk_ = Q.dot(s).dot(Q_inv)

# %%
plt.imshow(avk)
plt.colorbar()
plt.title('Original kernel')
plt.show()
# %%
plt.imshow(np.real(avk_))
plt.colorbar()
plt.title('Real part of reconstructed kernel')
plt.show()
2
# %%
diff = np.abs(avk - avk_)
plt.imshow(diff)
plt.colorbar()
plt.title('Difference between original and reconstructed kernel')
plt.show()

print(f'Mean error: {diff.mean()}')


#%%
