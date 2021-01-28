from io import BytesIO
from nibabel import FileHolder, Nifti1Image
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter
from time import time
from augmentations.elastic_deformation import elastic_transform
from augmentations.power_law_transformation import power_law_transformation

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