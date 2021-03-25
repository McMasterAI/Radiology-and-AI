import tqdm
import torchio as tio
import sys
import numpy as np
import pytorch_lightning as pl
import torchsummary
import torch
import os
sys.path.append('../MedicalZooPytorch')
from lib.medzoo.Unet3D import UNet3D
from lib.losses3D.basic import compute_per_channel_dice, expand_as_one_hot
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from collators.col_fn import col_fn

def run_eval(
    input_data_path,    
    model_path,
    validation_transform,
    output_results_path=None,
    input_channels_list = ['flair','t1','t2','t1ce'],
    input_image_dimensions = (240, 240, 160),
    seg_channels = [1,2,4],
    model_type = 'UNet3D',
    batch_size = 1,
    num_loading_cpus = 1,      
    train_val_split_ration=0.9,
    seed=42
):
    
    #DATASET CREATION
    subjects = []
    base_dir = input_data_path
    for file in tqdm.tqdm([file for file in os.listdir(base_dir) if os.path.isdir(base_dir + '/'+file) == True]):
        paths = [os.path.join(base_dir,file,file+f'_{chan}.nii.gz') for chan in input_channels_list]        
        subject = tio.Subject(            
            data = tio.ScalarImage(path = paths),
            seg = tio.LabelMap(path= [os.path.join(base_dir,file,file+'_seg.nii.gz')]),
            name = file
        )
        subjects.append(subject)        
    dataset = tio.SubjectsDataset(subjects)    
    
    #Splitting datasets into training and validation    
    training_split_ratio = train_val_split_ration
    num_subjects = len(dataset)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects
    num_split_subjects = num_training_subjects, num_validation_subjects
    generator=torch.Generator().manual_seed(seed)
    _, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects,generator)
    validation_set = tio.SubjectsDataset(validation_subjects, validation_transform)   
    
    model =  UNet3D(in_channels=len(input_channels_list), n_classes=len(seg_channels))
    model.load_state_dict(torch.load(model_path))
    model.eval()
                    
    dataloader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=num_loading_cpus,
        collate_fn=col_fn
    )

    losses = []
    with torch.no_grad():
        for i,batch in tqdm.tqdm(enumerate(dataloader)):
            x= batch['data']
            y = torch.cat([batch['seg'][:,x].unsqueeze(dim=1) for x in seg_channels],dim = 1)
            y_hat = model.forward(x)
            losses.append((-1*compute_per_channel_dice(y_hat, y)).detach().numpy().tolist())     
    #print(losses)
    for i in range(len(seg_channels)):
        avg = 0
        for j in range(len(losses)):
            avg += losses[j][i]
        print(avg/len(losses))
            