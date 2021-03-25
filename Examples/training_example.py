import sys
sys.path.append('./Radiology_and_AI')
from training.run_training import run_training
import torchio as tio

training_transform = tio.Compose([
    tio.ZNormalization(masking_method=tio.ZNormalization.mean), 
    tio.RandomBiasField(p=0.5),
    tio.CropOrPad((240, 240, 160)), 
    tio.OneHot(num_classes=5),
    
])

validation_transform = tio.Compose([
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.CropOrPad((240, 240, 160)),        
    tio.OneHot(num_classes=5)    
    
])

run_training(
    input_data_path = '../brats_new/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
    output_model_path = './Models/test_train_randbias_1e-3.pt',
    training_transform = training_transform,
    validation_transform = validation_transform,
    max_epochs=10,
    learning_rate = 1e-3,
    num_loading_cpus=2,
    batch_size = 2,
    train_val_split_ration=0.9,
    seed=42,
    amp_backend = 'apex',
    amp_level = 'O1',
    precision=16,
    wandb_logging = True,
    wandb_project_name = 'macai',
    wandb_run_name = 'randbias_1e-3',    
    
)