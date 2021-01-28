import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

# 23 second for function runs? This is barely custom code!
def elastic_transform(img_total, labels, alpha=1, sigma=2, c_val=0.0, method="linear"):

    i = img_total
    img_numpy = i[0]
    shape = img_numpy.shape
  
    # Define 3D coordinate system
    coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])
    # Interpolated img
    im_intrps = RegularGridInterpolator(coords, img_numpy,
                                        method=method,
                                        bounds_error=False,
                                        fill_value=c_val)

    # Get random elastic deformations
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha

    # Define sample points
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    indices = np.reshape(x + dx, (-1, 1)), \
              np.reshape(y + dy, (-1, 1)), \
              np.reshape(z + dz, (-1, 1))

    # Interpolate 3D image image
    i[0] = im_intrps(indices).reshape(shape)

    for channel in range(1, img_total.shape[0]):
      img_numpy = i[channel]
      shape = img_numpy.shape
      # Interpolated img
      im_intrps = RegularGridInterpolator(coords, img_numpy,
                                          method=method,
                                          bounds_error=False,
                                          fill_value=c_val)
      i[channel] = im_intrps(indices).reshape(shape)


    lab_intrp = RegularGridInterpolator(coords, labels,
                                        method="nearest",
                                        bounds_error=False,
                                        fill_value=0)

    labels = lab_intrp(indices).reshape(shape).astype(labels.dtype)

    return i, labels
