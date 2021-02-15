from io import BytesIO
from nibabel import FileHolder, Nifti1Image
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter
from time import time
from random import random
from augmentations.elastic_deformation import elastic_transform
from augmentations.power_law_transformation import power_law_transformation
from scipy.interpolate import interp1d

def col_img(batch, to_tensor = True, nyul_params = None, use_zscore= False, pl_prob = 0, elastic_prob=0):
    """
    Collator function for dataloader. When putting this into the dataloader, use a lambda function(batch) with the parameters you need.
    Args:
        batch: not sure, but entirely handled by the DataLoader itself.
        to_tensor (bool): If False, then return NumPy array. If True, return torch tensors.
        nyul_params (dict): dict containing keys "percs" and "standard_scales" from nyul functions
        use_zscore (bool): If True and nyul_params is None, then use Z-score normalization.
        pl_prob (float): Probability of doing power law transformation
        elastic_prob (float): Not currently available, but would be probability of doing elastic deformation.
    """
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
    f_flair = np.expand_dims(np.pad(f_flair, padding), axis=0)
    f_t1 = np.expand_dims(np.pad(f_t1, padding), axis=0)
    f_t2 = np.expand_dims(np.pad(f_t2, padding), axis=0)
    f_t1ce = np.expand_dims(np.pad(f_t1ce, padding), axis=0)
    f_seg = np.pad(f_seg, padding)
    concat = np.concatenate([f_t1, f_t1ce, f_t2, f_flair], axis=0)
    f_seg = np.expand_dims(f_seg, axis=0)

    assert not ((nyul_params is not None) and (use_zscore))
    
    # Normalizations
    if nyul_params is not None:
      percss = nyul_params['percs']
      standard_scales = nyul_params['standard_scales']
      for i in range(concat.shape[0]):
        concat[i] = dataloader_hist_norm(concat[i], percss[i], standard_scales[i], f_seg, ignore_zero=True)/100


    if use_zscore:
        for i in range(concat.shape[0]):
          concat[i] = Zscore_normalize(concat[i], floor=-3)
   

    # Augmentations - Elastic transform not implemented (quickly) yet

    if random() < pl_prob:
      concat, f_seg = power_law_transformation(concat, f_seg) 

    assert elastic_prob == 0
    if random() < elastic_prob:
      concat, f_seg = elastic_transform(concat, f_seg, sigma=5)

    if to_tensor:
      concat = torch.as_tensor(concat).half()
      f_seg = torch.as_tensor(f_seg) 

    return ([concat, f_seg])
def nyul_train_dataloader(dataloader, n_imgs = 4, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10, ignore_zero=True):
    """
    determine the standard scale for the set of images
    Args:
        img_fns (list): set of NifTI MR image paths which are to be normalized
        mask_fns (list): set of corresponding masks (if not provided, estimated)
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)
    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    percss = [np.concatenate(([i_min], np.arange(l_percentile, u_percentile+1, step), [i_max])) for _ in range(n_imgs)]
    standard_scales = [np.zeros(len(percss[0])) for _ in range(n_imgs)]

    iteration = 1
    for all_img, seg_data in dataloader:
      print(iteration)
    
     # print(seg_data.shape)
      mask_data = seg_data
      if ignore_zero:
        mask_data[seg_data ==0] = 1

      mask_data = np.squeeze(mask_data, axis=0)

      #mask_data[mask_data==2] = 0 # ignore edema

      for i in range(n_imgs):
        img_data = all_img[i]
        if ignore_zero:
          masked = img_data[mask_data > 0]
        else:
          masked = img_data

        landmarks = intensity_normalization.normalize.nyul.get_landmarks(masked, percss[i])
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])
        landmarks = np.array(f(landmarks))
        
        standard_scales[i] += landmarks
      iteration += 1

    standard_scales = [scale / iteration for scale in standard_scales]
    return standard_scales, percss
  
def dataloader_hist_norm(img_data, landmark_percs, standard_scale, seg_data, ignore_zero = False):
    """
    do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks
    Args:
        img (nibabel.nifti1.Nifti1Image): image on which to find landmarks
        landmark_percs (np.ndarray): corresponding landmark points of standard scale
        standard_scale (np.ndarray): landmarks on the standard scale
        mask (nibabel.nifti1.Nifti1Image): foreground mask for img
    Returns:
        normalized (nibabel.nifti1.Nifti1Image): normalized image
    """
    mask_data = seg_data
    if ignore_zero:
      mask_data[seg_data ==0] = 1
    mask_data = np.squeeze(mask_data, axis=0)

    masked = img_data[mask_data > 0]

    landmarks = intensity_normalization.normalize.nyul.get_landmarks(masked, landmark_percs)
    f = interp1d(landmarks, standard_scale, fill_value='extrapolate')
    normed = f(img_data)

    z = img_data
    if ignore_zero:
      z[img_data > 0] = normed[img_data > 0]
    return z #normed

def Zscore_normalize(img, floor=None):
  img_above_0 = np.ravel(img)
  img_above_0 = img_above_0[img_above_0>0]
  mean = np.mean(img_above_0)
  sd = np.std(img_above_0)
  if floor is None:
    return (img - mean)/sd
  else:

    zero_mask = (img==0)

    i = (img - mean)/sd
    i = i - floor
    i[i < 0] = 0
    i[zero_mask] = 0

    return i    