import tqdm
import torchio as tio
import sys
import numpy as np
import pytorch_lightning as pl
import torchsummary
import torch
sys.path.append('../MedicalZooPytorch')
from lib.medzoo.Unet3D import UNet3D
from lib.losses3D.basic import compute_per_channel_dice, expand_as_one_hot
from torch.utils.data import Dataset, DataLoader, random_split
from lightning_modules.TumourSegmentation import TumourSegmentation
from collators.col_fn import col_fn
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import os

def run_training(
    input_data_path,
    output_model_path,
    training_transform,
    validation_transform,
    input_channels_list = ['flair','t1','t2','t1ce'],
    seg_channels = [1,2,4],
    training_split_ratio = 0.9,
    model_type = 'UNet3D',
    batch_size = 1,
    num_loading_cpus = 1,
    learning_rate = 1e-3,
    wandb_logging = False,
    wandb_project_name = None,
    wandb_run_name = None,
    seed=42,
    
    accumulate_grad_batches = 1,
    default_root_dir='./Models/checkpoints'
    gpus=1,
    max_epochs = 10,
    precision=16,
    check_val_every_n_epoch = 1,
    log_every_n_steps=10,      
    val_check_interval= 50,
    progress_bar_refresh_rate=1,    
    **kwargs
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
    num_subjects = len(dataset)
    print('Num Subjects: ',num_subjects)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects
    num_split_subjects = num_training_subjects, num_validation_subjects
    generator=torch.Generator().manual_seed(seed)
    training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects,generator)
    training_set = tio.SubjectsDataset(training_subjects, training_transform)
    validation_set = tio.SubjectsDataset(validation_subjects, validation_transform)
    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')
    
    #TRAINER RUNNING
    wandb_logger = None
    if wandb_logging:
        wandb_logger = WandbLogger(project=wandb_project_name,name=wandb_run_name, offline = False)
        
    model = TumourSegmentation(
        train_dataset=training_set,
        val_dataset=validation_set,
        col_fn=col_fn,
        batch_size=batch_size,
        num_loading_cpus=num_loading_cpus,
        in_channels=len(input_channels_list),
        classes=seg_channels,
        learning_rate=learning_rate
    )
    
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        accumulate_grad_batches=accumulate_grad_batches,
        gpus=gpus,
        max_epochs=max_epochs,
        precision=precision,
        check_val_every_n_epoch=check_val_every_n_epoch,        
        log_every_n_steps=log_every_n_steps,      
        val_check_interval=val_check_interval,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        logger = wandb_logger,
        **kwargs
    )
    trainer.fit(model)
    
    #OUTPUT
    torch.save(model.model.state_dict(), output_model_path)