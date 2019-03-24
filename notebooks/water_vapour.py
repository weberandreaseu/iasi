# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from netCDF4 import Dataset

import matplotlib
plt.switch_backend('PDF')
matplotlib.use('PDF')

# plt.rc('font', family='Linux Libertine')
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Modern Computer'], 'serif':['Linux Libertine']})
# plt.rc('text', usetex=True)

nc = Dataset('test/resources/MOTIV-single-event.nc')
avk = nc['state_WVatm_avk'][0]
xavk = nc['state_Tatm2WVatm_xavk'][0]
atm_n = nc['state_WVatm_n'][0]

level = 28
interpolation = None

fig = plt.figure(figsize=(20, 6))
# fig.suptitle('Water Vapour')

gs = fig.add_gridspec(nrows=2, ncols=5)


def create_image_grid(grid_spec, nrows_ncols=(2, 2)):
    return ImageGrid(fig,
                     grid_spec,
                     nrows_ncols=nrows_ncols,
                     axes_pad=0.1,  # pad between axes in inch.
                     cbar_mode='single',
                     cbar_location="right",
                     cbar_size="5%",
                     cbar_pad=0.05,
                     )


def plot_image_grid(matrix, title, position):
    grid = create_image_grid(position)
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            im = grid[idx].imshow(
                matrix[j, i, :level, :level], interpolation=interpolation)
            grid[idx].cax.colorbar(im)
    grid[0].set_ylabel("H2O")
    grid[2].set_ylabel("HDO")
    grid[2].set_xlabel("H2O")
    grid[3].set_xlabel("HDO")
    grid[0].set_title(title, x=1)


# averaging kernel
plot_image_grid(avk, 'Averaging Kernel', gs[:2, :2])
# noise matrix
plot_image_grid(atm_n, 'Noise Matrix', gs[:2, 2:4])
# cross averaging kernel
grid = create_image_grid(gs[:, 4:5], (2, 1))
for i in range(2):
    im = grid[i].imshow(xavk[i, :level, :level], interpolation=interpolation)
    grid[i].cax.colorbar(im)

grid[0].set_title('Cross Averaging Kernel')
grid[0].set_ylabel("H2O")
grid[1].set_ylabel("HDO")
grid[1].set_xlabel("Temperature")

# plt.savefig('water_vapour.pdf', bbox_inches='tight')
plt.show()
