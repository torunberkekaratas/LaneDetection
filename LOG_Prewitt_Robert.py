"""
==============
Edge Detection
==============

Roberts, Laplace of Gaussian and Prewitt Edge Detection are implemented in this code.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.data import camera
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, \
    scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h, farid_v, farid_h, laplace, gaussian, prewitt

img = io.imread("test_image.jpg");
image = img[:,:,0];

# Roberts Edge Detection
edge_roberts = roberts(image);

# Laplace of Gaussiann(LOG) Edge Detectipn
edge_LOG = laplace(gaussian(image));

# Horizontal Prewitt
edge_prewitt_h = prewitt_h(image);

# Vertical Prewitt
edge_prewitt_v = prewitt_v(image);

# Prewitt Edge Detection
edge_prewitt = prewitt(image);



# Plot Images
fig, ax = plt.subplots(ncols=6, sharex=True, sharey=True,
                       figsize=(12, 6))


ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original Image')

ax[1].imshow(edge_roberts, cmap=plt.cm.gray)
ax[1].set_title('Roberts Edge Detection')

ax[2].imshow(edge_LOG, cmap=plt.cm.gray)
ax[2].set_title('LOG Edge Detection')

ax[3].imshow(edge_prewitt_h, cmap=plt.cm.gray)
ax[3].set_title('Prewitt_h Edge Detection')

ax[4].imshow(edge_prewitt_v, cmap=plt.cm.gray)
ax[4].set_title('Prewitt_v Edge Detection')


ax[5].imshow(edge_prewitt, cmap=plt.cm.gray)
ax[5].set_title('Prewitt Edge Detection')


for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
