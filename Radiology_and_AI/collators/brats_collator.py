from io import BytesIO
from nibabel import FileHolder, Nifti1Image
import torch
import numpy as np

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

    f_flair = torch.as_tensor(np.expand_dims(np.pad(f_flair, [(0, 0), (0, 0), (2, 3)]), axis=0)/1000).half()
    f_t1 = torch.as_tensor(np.expand_dims(np.pad(f_t1, [(0, 0), (0, 0), (2, 3)]), axis=0)/1000).half()
    f_t2 = torch.as_tensor(np.expand_dims(np.pad(f_t2, [(0, 0), (0, 0), (2, 3)]), axis=0)/1000).half()
    f_t1ce = torch.as_tensor(np.expand_dims(np.pad(f_t1ce, [(0, 0), (0, 0), (2, 3)]), axis=0)/1000).half()

    orig_f_seg = np.expand_dims(np.pad(f_seg, [(0, 0), (0, 0), (2, 3)]), axis=0)
    concat = torch.cat([f_t1, f_t1ce, f_t2, f_flair], axis=0)
    f_seg = torch.as_tensor(orig_f_seg) #.half()

    # remove nan?
    f_seg[f_seg != f_seg] = 0
    concat[concat != concat] = 0

    return ([concat, f_seg])

def selector_train(x):
  folder_name = list(x.items())[0][1].split('/')[-1]
  if len(folder_name) >= 17:
    return int(folder_name[17]) < 3
  else:
    return False
  
def selector_eval(x):
  folder_name = list(x.items())[0][1].split('/')[-1]
  if len(folder_name) >= 17:
    return int(folder_name[17]) == 3
  else:
    return False
  
  
