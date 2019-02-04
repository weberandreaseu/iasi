# %%
import luigi
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from sklearn.metrics import mean_absolute_error

from iasi import EigenDecomposition, SingularValueDecomposition

# %%
dim = 10
file = 'test/resources/IASI-test-single-event.nc'


def get_first(file, variable):
    nc = Dataset(file)
    return nc[variable][0, 0, 0]


def get_block(file, variable):
    nc = Dataset(file)
    event_var = nc[variable][0]
    return np.block(
        [[event_var[0, 0], event_var[1, 0]],
         [event_var[0, 1], event_var[1, 1]]]
    )


def plot_matrix(matrix: np.ndarray, tile: None):
    plt.imshow(matrix)
    plt.colorbar()
    plt.title(tile)
    plt.show()


def eigen_composition(path):
    dataset = Dataset(path)
    Q = dataset['state_WVatm_avk_Q'][0, 0, 0]
    s = dataset['state_WVatm_avk_s'][0, 0, 0]
    Q_inv = np.linalg.pinv(Q)
    s = np.diag(s)
    return Q.dot(s).dot(Q_inv)

def sv_composition(path):
    dataset = Dataset(path)
    U = dataset['state_WVatm_avk_U'][0, 0, 0]
    s = dataset['state_WVatm_avk_s'][0, 0, 0]
    Vh = dataset['state_WVatm_avk_Vh'][0, 0, 0]
    sigma = np.diag(s)
    return np.dot(U, np.dot(sigma, Vh))

# %%
eigen_decomposition = EigenDecomposition(
    dim=dim,
    file='test/resources/IASI-test-single-event.nc',
    dst='./data'
)
luigi.build([eigen_decomposition], local_scheduler=True)
# %%
sv_decomposition = SingularValueDecomposition(
    dim=dim,
    file='test/resources/IASI-test-single-event.nc',
    dst='./data'
)
luigi.build([sv_decomposition], local_scheduler=True)


# %%
avk_original = get_first(file, 'state_WVatm_avk')
avk_eigen = eigen_composition(eigen_decomposition.output().path)
avk_svd = sv_composition(sv_decomposition.output().path)

#%%
plot_matrix(avk_original, 'Original AVK')
plot_matrix(avk_eigen, 'AVK from eigen decomposition')
plot_matrix(avk_svd, 'AVK from sv decomposition')

#%%
plot_matrix(np.abs(avk_eigen - avk_original), 'Difference between original kernel and eigen decomposition')
plot_matrix(np.abs(avk_svd - avk_original), 'Difference between original kernel and sv decomposition')

print('Mean absolute error')
print(f'Eigen decomposition: {mean_absolute_error(avk_original, avk_eigen)}')
print(f'SV decomposition: {mean_absolute_error(avk_original, avk_svd)}')
