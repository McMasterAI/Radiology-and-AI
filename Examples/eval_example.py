import sys
sys.path.append('./Radiology_and_AI')
from validation.run_eval import run_eval
import torchio as tio

validation_transform = tio.Compose([
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.CropOrPad((240, 240, 160)),        
    tio.OneHot(num_classes=5)    
    
])

run_eval(
    input_data_path= '../brats_new/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
    model_path="../randgamma.pt",
    validation_transform=validation_transform,
    seed=42,
    train_val_split_ration=0.9,
    batch_size=1
)