import torch
import tqdm
import torchio as tio
import sys
sys.path.append('../MedicalZooPytorch')
sys.path.append('./Radiology_and_AI')
import numpy
import os
from lib.medzoo.Unet3D import UNet3D
from collators.col_fn import col_fn
from visuals.gif_functions import create_images, make_gif
from visuals.slice_functions import plot_volume

def gen_visuals(
    image_path,
    transforms, #Apply whatever transforms were also applied to the validation set during training i.e the normalization and data preparation transforms, not augmentation
    true_gif_output_path=None, #WHere to put the generated gif images for your outputted gif of the brain with true segmentation labels
    pred_gif_output_path=None, #WHere to put the generated gif images for your outputted gif of the brain with predicted segmentation labels  
    model_path=None, #The path to the model which will be used for outputting predicted segmentations
    input_channels_list=['flair','t1','t2','t1ce'], #The names of the different modalities your input example folder containts
    seg_channels=[1,2,4], #which segmentation values your input image has
    seg_channels_to_display_gif=[1,2,4], #which segmentation values to display, max of three
    gen_gif=True, #Output a gif of the brain with or without segmetnation and  at angles determined by gif_view_angle and gif_angle_rotation    
    gen_pred=True, #Generate output using the predicted segmentation values
    gen_true=True, #Generate output using the true segmentation values
    gif_view_angle=30, #The vertical angle your gif will "look down" on the rotating brain at
    gif_angle_rotation=20,  #How many degress to rotate the brain between gif images
    fig_size_gif = (50,25),
    
    slice_output_path=None, #Where to output the generated slics​ of the input example
    gen_slice=False, #Output a slice of the brain at a specific dimension determined by slice_dimension and slice_num
    fig_size_slice = (25,50),
    seg_channels_to_display_slice = [1,2,4],
    sag_slice = None,
    cor_slice = None,
    axi_slice = None,
    disp_slice_base = True,
    slice_title = None,
    
    gen_nifti=False,
    nifti_output_path=None,
    
):

    subjects = []
    folder = image_path.split('/')[-1]
    paths = [os.path.join(image_path,folder+f'_{chan}.nii.gz') for chan in input_channels_list]
    if gen_true:
        subject = tio.Subject(        
            data = tio.ScalarImage(path = paths),
            seg = tio.LabelMap(path=[image_path  +'/'+ folder+ '_seg.nii.gz']),
            name = folder
        )
    else:
        subject = tio.Subject(        
            data = tio.ScalarImage(path = paths),        
            name = folder
        )
        
    subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects,transforms)
    
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True,collate_fn=col_fn) 
    if gen_pred:
        model = UNet3D(in_channels=len(input_channels_list), n_classes=len(seg_channels))
        model.load_state_dict(torch.load(model_path))
        model.eval()
    for subject in val_dataloader:
        if gen_pred:
            prediction = torch.nn.Sigmoid()(model(subject['data'])).detach()
        else:
            pass
        
    if gen_nifti:
        mri = tio.ScalarImage(tensor=subject['data'][0]) 
        mri.save(os.path.join(nifti_output_path,'stacked.nii.gz'))        
      
    if gen_pred:
        channels = [] 
        for i in seg_channels_to_display_gif:            
            channels.append(prediction[0][seg_channels.index(i)] > 0.5)
        if gen_slice:        
            plot_volume(
                prediction[0].numpy()>0.5,
                base = subject['data'][0].numpy(),
                disp_base = disp_slice_base,
                all_segs=seg_channels,
                disp_segs=seg_channels_to_display_slice,
                fig_size=fig_size_slice,
                title = slice_title,
                sag_slice = sag_slice,
                cor_slice = cor_slice,
                axi_slice = axi_slice,
                save_path= os.path.join(slice_output_path,'pred.png')
            )
        if gen_nifti:
            adj_seg = tio.LabelMap(tensor=prediction[0].numpy() > 0.5)
            adj_seg.save(os.path.join(nifti_output_path,'pred_seg.nii.gz'))              
        if gen_gif:
            channels += [numpy.array([None])] * (3 - len(channels))
            create_images(pred_gif_output_path, channels[0], channels[1], channels[2], mri_chan = subject['data'][0][1] > 0,angle_num = gif_angle_rotation,angle_view=gif_view_angle,fig_size =fig_size_gif)         
            make_gif(pred_gif_output_path,os.path.join(pred_gif_output_path,'pred.gif'),angle_num=gif_angle_rotation)        
       
            
    if gen_true:
        channels = [] 
        for i in seg_channels_to_display_gif:            
            channels.append(subject['seg'][0][i] > 0.5)
        if gen_slice: 
            plot_volume(
                subject['seg'][0,seg_channels].numpy(),
                base = subject['data'][0].numpy(),
                disp_base = disp_slice_base,
                all_segs=seg_channels,
                disp_segs=seg_channels_to_display_slice,
                fig_size=fig_size_slice,
                title = slice_title,
                sag_slice = sag_slice,
                cor_slice = cor_slice,
                axi_slice = axi_slice,
                save_path= os.path.join(slice_output_path,'true.png')
            )
        if gen_nifti:
            adj_seg = tio.LabelMap(tensor=subject['seg'][0, seg_channels] > 0.5)
            adj_seg.save(os.path.join(nifti_output_path,'true_seg.nii.gz'))            
        if gen_gif:
            channels += [numpy.array([None])] * (3 - len(channels))        
            create_images(true_gif_output_path, channels[0], channels[1], channels[2], mri_chan = subject['data'][0][1] > 0,angle_num = gif_angle_rotation,angle_view=gif_view_angle,fig_size =fig_size_gif) 
            make_gif(true_gif_output_path,os.path.join(true_gif_output_path,'true.gif'),angle_num=gif_angle_rotation)
        


    
if __name__ == "__main__":    
    validation_transform = tio.Compose([
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.CropOrPad((240, 240, 160)),        
        tio.OneHot(num_classes=5)    

    ])        
    gen_visuals(
        image_path="../brats_new/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_010",
        transforms = validation_transform,
        model_path =  "./Models/test_train_many_1e-3.pt",
        gen_pred = True,
        gen_true = True,
        input_channels_list = ['flair','t1','t2','t1ce'],
        seg_channels = [1,2,4],
        
        gen_gif = False,
        true_gif_output_path = "../output/true",
        pred_gif_output_path = "../output/pred",     
        seg_channels_to_display_gif = [1,2,4],
        gif_view_angle = 30,
        gif_angle_rotation = 20,
        fig_size_gif = (50,25),
        
        gen_slice = True,
        slice_output_path = "../output/slices",
        fig_size_slice = (25,50),
        seg_channels_to_display_slice = [2,4,1],
        sag_slice = None,
        cor_slice = None,
        axi_slice = None,
        disp_slice_base = True,
        slice_title = None,
        
        gen_nifti = True,
        nifti_output_path = "../output/nifti",
    )