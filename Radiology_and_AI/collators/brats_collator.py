from io import BytesIO
from nibabel import FileHolder, Nifti1Image
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter
from time import time

def col_img(batch):
    bytes_data_list = [list(batch[i].items())[1][1] for i in range(5)]
    bytes_data_keys = [list(batch[i].items())[0][1].split('_')[-1] for i in range(5)]
    bytes_data_dict = dict(zip(bytes_data_keys,bytes_data_list))

    bb = BytesIO(bytes_data_dict['flair'])
    fh = FileHolder(fileobj=bb)
    f_flair = Nifti1Image.from_file_map({'header': fh, 'image':fh}).get_fdata()
    bb = BytesIO(bytes_data_dict['seg'])
    fh = FileHolder(fileobj=bb)
    f_seg = Nifti1Image.from_file_map({'header': fh, 'image':fh}).get_fdata()
    bb = BytesIO(bytes_data_dict['t1'])
    fh = FileHolder(fileobj=bb)
    f_t1 = Nifti1Image.from_file_map({'header': fh, 'image':fh}).get_fdata()
    bb = BytesIO(bytes_data_dict['t1ce'])
    fh = FileHolder(fileobj=bb)
    f_t1ce=Nifti1Image.from_file_map({'header':fh, 'image':fh}).get_fdata()
    bb = BytesIO(bytes_data_dict['t2'])
    fh = FileHolder(fileobj=bb)
    f_t2 =Nifti1Image.from_file_map({'header':fh, 'image':fh}).get_fdata()

    padding = [(0, 0), (0, 0), (2, 3)]
    f_flair = torch.as_tensor(np.expand_dims(np.pad(f_flair, padding), axis=0)/1000).half()
    f_t1 = torch.as_tensor(np.expand_dims(np.pad(f_t1, padding), axis=0)/1000).half()
    f_t2 = torch.as_tensor(np.expand_dims(np.pad(f_t2, padding), axis=0)/1000).half()
    f_t1ce = torch.as_tensor(np.expand_dims(np.pad(f_t1ce, padding), axis=0)/1000).half()

    f_seg = np.pad(f_seg, padding)
    concat = np.concatenate([f_t1, f_t1ce, f_t2, f_flair], axis=0)

    concat, f_seg = power_law_transformation(concat, f_seg) 
    #concat, f_seg = elastic_transform(concat, f_seg, sigma=5)
    f_seg = np.expand_dims(f_seg, axis=0)
    


    concat = torch.as_tensor(concat).half()
    f_seg = torch.as_tensor(f_seg) 

    # remove nan?
    f_seg[f_seg != f_seg] = 0
    concat[concat != concat] = 0

    return ([concat, f_seg])
  
  
# This probably works as intended
def _pl(img, gain, gamma):
  return img**gamma * gain

# This is about 1.2 seconds per iteration, which is probably acceptable
def power_law_transformation(img, seg, gain_distr=[0.8, 1.2], gamma_distr=[0.8, 1.2]):
  # Apply this scaling to all images in series (as opposed to each independently?)
  i = img
  gain = np.random.uniform(low=gain_distr[0], high=gain_distr[1])
  gamma = np.random.uniform(low=gamma_distr[0], high=gamma_distr[1])
  #for channel in i.shape[0]:
  #  i[channel] = _pl(i[channel], gain, gamma)
  i = _pl(i, gain, gamma)
  return i, seg

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

