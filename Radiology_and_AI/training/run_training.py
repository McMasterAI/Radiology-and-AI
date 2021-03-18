import tqdm
import torchio as tio
import sys
import numpy as np
import pytorch_lightning as pl
import torchsummary
sys.path.append('./MedicalZooPytorch')
from lib.medzoo.Unet3D import UNet3D
from lib.losses3D.basic import compute_per_channel_dice, expand_as_one_hot
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import os

def run_training(
    input_data_path,
    output_model_path,
    trainer,
    training_transform,
    validation_transform,
    input_channels_list = ['flair','t1','t2','t1ce'],
    input_image_dimensions = (240, 240, 160),
    seg_channels = [1,2,4],
    train_val_split_ration = 0.9,
    model_type = 'UNet3D',
    batch_size = 1,
    num_loading_cpus = 1,
    learning_rate = 5e-5,
    wandb_logging = False,
    wandb_project_name = None,
    wandb_run_name = None,
    
    accumulate_grad_batches = 1,
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
    for file in tqdm.tqdm([file for file in os.listdir('./brats_new/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData') if os.path.isdir(base_dir + file) == True]):
        subject = tio.Subject(
            paths = [os.path.join(base_dir,file,file+f'_{chan}.nii.gz') for chan in input_channels_list]
            data = tio.ScalarImage(path = paths),
            seg = tio.LabelMap(path= [os.path.join(base_dir,file,file+'_seg.nii.gz')])
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
    training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)
    training_set = tio.SubjectsDataset(training_subjects, training_transform)
    validation_set = tio.SubjectsDataset(validation_subjects, validation_transform)
    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')
    
    #TRAINER RUNNING
    wandb_logger = None
    if wandb_logging:
        wandb_logger = WandbLogger(project=wandb_project_name,name=wandb_run_name, offline = False)
        
    model = TumourSegmentation(training_set,validation_set,col_fn,batch_size,num_loading_cpus,seg_channels,learning_rate)
    
    trainer = pl.Trainer(
        accumulate_grad_batches,
        gpus,
        max_epochs,
        precision,
        check_val_every_n_epoch,
        logger = wandb_logger,
        log_every_n_steps,      
        val_check_interval,
        progress_bar_refresh_rate,
        **kwargs
    )
    trainer.fit(model)
    
    #OUTPUT
    torch.save(model.model.state_dict(), output_model_path)