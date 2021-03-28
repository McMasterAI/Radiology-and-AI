from typing import Optional,Tuple, List
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

#Heavily modified from https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
#Used for generating slices image and segmentation
def plot_volume(
        array: np.ndarray,
        disp_base: bool,
        all_segs: List[int],
        disp_segs: List[int],
        save_path: str,
        base: Optional[np.ndarray] = None,
        title: Optional[str] = None,        
        sag_slice: Optional[int] = None,
        cor_slice: Optional[int] = None,
        axi_slice: Optional[int] = None,
        fig_size : Tuple[int] = (25, 50),    
        ):
    
    #Dimensions of image
    si, sj, sk = array.shape[1:4]
    #Slices of images to take
    i =  sag_slice if sag_slice is not None else si // 2
    j =  cor_slice if cor_slice is not None else sj // 2
    k =  axi_slice if axi_slice is not None else sk // 2
    #Adding all segmentation layers we specified
    slices=[[],[],[]]
    for seg in disp_segs:
        slices[0].append(array[all_segs.index(seg),i, :,:])
        slices[1].append(array[all_segs.index(seg),:, j, :])
        slices[2].append(array[all_segs.index(seg),:, :, k])
    #Adding slices to display the mri scan itself as background
    base_slices=[[],[],[]]
    if disp_base:
        for seg in range(base.shape[0]):
            base_slices[0].append(base[seg,i, ...])
            base_slices[1].append(base[seg,:, j, ...])
            base_slices[2].append(base[seg,:, :, k, ...])
            break
    #The colours were using to display the background and seg
    cmap = ['gray','summer' ,'autumn', 'winter']
    labels = 'AS', 'RS', 'RA'
    titles = 'Sagittal', 'Coronal', 'Axial'
    
    #Displaying three subplot, one for each view of the brain
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 3, width_ratios=[256 / 160, 1, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    axes = ax1, ax2, ax3
    #Displaying all layers (base plus segmentations) for  each of the three subpplots
    for (base_slice_ ,slice_, axis, label, stitle) in zip(base_slices, slices, axes, labels, titles):
        if len(base_slice_) > 0: #display background
            axis.imshow(turn(base_slice_[0]), cmap=cmap[0]) 
        for c_ind,layer in enumerate(slice_): #display all segmentation layers
            axis.imshow(turn(np.ma.masked_where(layer == 0, layer)), alpha=0.6,cmap=cmap[c_ind+1],interpolation='none')
        axis.grid(False)
        axis.invert_xaxis()
        axis.invert_yaxis()
        x, y = label
        axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.set_title(stitle)
        axis.set_aspect('equal')
    if title is not None:
        plt.gcf().suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def turn(array_2d: np.ndarray) -> np.ndarray:
    return np.flipud(np.rot90(array_2d))    
 