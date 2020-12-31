from __future__ import absolute_import, division, print_function

import csv
import json
import os
import math
import random
import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import nibabel as nb
import numpy as np
import random
import gc

class BratsDatasetConfig(datasets.BuilderConfig):
    def __init__(self, data_folder,data_version,**kwargs):
        self.data_path =data_folder
        self.data_version =data_version
        

class BratsDataset(datasets.GeneratorBasedBuilder):
    _writer_batch_size = 1
    BUILDER_CONFIG_CLASS = BratsDatasetConfig
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "channels": datasets.Array4D(shape=(4,240,240,155),dtype='float32'),                    
                    "segmentation": datasets.Array3D(shape=(240,240,155),dtype='float32')
                }
            ),
        )

    def _split_generators(self,dl_manager):
        """Returns SplitGenerators."""
        data_files = []
        for i in range(1,355):
            i = str(i).rjust(3,'0')
            print(i, end=" ")
            data_files.append(os.path.join(self.config.data_path,f'BraTS20_Training_{i}/BraTS20_Training_{i}_flair.nii'))            
            
        for i in range(356,370):
            i = str(i).rjust(3,'0')
            print(i, end=" ")
            data_files.append(os.path.join(self.config.data_path,f'BraTS20_Training_{i}/BraTS20_Training_{i}_flair.nii'))            
                        
        urls_to_download = {
                "main": data_files
        }                
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["main"],"split": "train"}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["main"],"split": "validation"})
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """        
        #ex_list = []
        
        train_ex, val_ex = train_test_split(filepath, test_size = 0.20, random_state = 1)
        
        if split == 'train':
            ex_list = train_ex
        elif split == 'validation':           
            ex_list = val_ex          
            
        for id_, file in enumerate(ex_list):
            file = "_".join(file.split('_')[:-1])
            f_flair = nb.load(os.path.join(file+ '_flair.nii')).get_fdata()
            f_seg = nb.load(os.path.join(file+'_seg.nii')).get_fdata()
            f_t1ce = nb.load(os.path.join(file+'_t1ce.nii')).get_fdata()
            f_t1 =  nb.load(os.path.join(file+'_t1.nii')).get_fdata()
            f_t2 = nb.load(os.path.join(file+'_t2.nii')).get_fdata()
            ex = [np.stack([f_t1, f_t1ce, f_t2, f_flair]), f_seg]       
            yield id_, {
                "channels": ex[0],
                "segmentation": ex[1]
            }                   