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

def run_inference(
    input_data_path,
    output_results_path,
    model_path,
    test_transform,
    input_channels_list = ['flair','t1','t2','t1ce'],
    input_image_dimensions = (240, 240, 160),
    seg_channels = [1,2,4],
    model_type = 'UNet3D',
    batch_size = 1,
    num_loading_cpus = 1,       
):