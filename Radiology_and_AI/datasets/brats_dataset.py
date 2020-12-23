import torch
import nibabel as nb
import os
import numpy as np

#Dataset
class brats_dataset(torch.utils.data.Dataset):
  def __init__(self,data_folders):
    self.data_list = []
    
    #Perform necessary input data preparation in this function
    #add each input example into the data_last function
    #takes in a list of folders and processes the data contained

    # U net requires all dimensions be divisible by 8 (by default)
    # or we'd have to manually do the padding in the U-net model
    # no padding="valid" exists in Pytorch for... reasons?
    for i, folder in enumerate(data_folders):
      i_str = folder[-3:]

      f_flair = nb.load(os.path.join(folder,'BraTS20_Training_%s_flair.nii' % i_str),mmap=False).get_fdata()
      f_seg = nb.load(os.path.join(folder,'BraTS20_Training_%s_seg.nii'% i_str),mmap=False).get_fdata()
      f_t1ce = nb.load(os.path.join(folder,'BraTS20_Training_%s_t1ce.nii'% i_str),mmap=False).get_fdata()
      f_t1 =  nb.load(os.path.join(folder,'BraTS20_Training_%s_t1.nii'% i_str),mmap=False).get_fdata() 
      f_t2 = nb.load(os.path.join(folder,'BraTS20_Training_%s_t2.nii'% i_str),mmap=False).get_fdata()

      f_flair = torch.as_tensor(np.expand_dims(np.pad(f_flair, [(0, 0), (0, 0), (2, 3)]), axis=0)).half()
      f_t1 = torch.as_tensor(np.expand_dims(np.pad(f_t1, [(0, 0), (0, 0), (2, 3)]), axis=0)).half()
      f_t2 = torch.as_tensor(np.expand_dims(np.pad(f_t2, [(0, 0), (0, 0), (2, 3)]), axis=0)).half()
      f_seg = torch.as_tensor(np.expand_dims(np.pad(f_seg, [(0, 0), (0, 0), (2, 3)]), axis=0)).half()
      f_t1ce = torch.as_tensor(np.expand_dims(np.pad(f_t1ce, [(0, 0), (0, 0), (2, 3)]), axis=0)).half()


      concat = torch.cat([f_t1, f_t1ce, f_t2, f_flair], axis=0)

      self.data_list.append([concat, f_seg])
  def __len__(self):
    return len(self.data_list)
  def __getitem__(self, index):
    return self.data_list[index]