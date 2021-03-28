from mpl_toolkits.mplot3d import Axes3D
import numpy
import os
import imageio
import matplotlib.pyplot as plt

#Creates images at specified angle intervals using voxels to display in 3D
#Downscales original resolution of input
def create_images(out_dir,seg_1= numpy.array([None]),seg_2 = numpy.array([None]), seg_3 = numpy.array([None]), mri_chan=numpy.array([None]),downsize_factor=5,angle_num = 20,angle_view= 30,fig_size=(50,25)): 
    for angle in range(0, 360, angle_num):
        ax = make_ax(True,fig_size=fig_size)
        if (seg_1.any() == None) == False: #core        
            seg_1_small = make_smaller(seg_1,downsize_factor)
            ax.voxels(seg_1_small, facecolors='#FF000080', edgecolors='gray', shade=False)
        if (seg_2.any() == None) == False: #enhancing
            seg_2_small = make_smaller(seg_2,downsize_factor)
            ax.voxels(seg_2_small, facecolors='#0000FF40', shade=False) 
        if (seg_3.any() == None) == False: # edema
            seg_3_small = make_smaller(seg_3,downsize_factor)
            ax.voxels(seg_3_small, facecolors='#00FF0020', shade=False) 

        if (mri_chan.any() == None) == False:
            mri_chan_small = make_smaller(mri_chan,downsize_factor) 
            ax.voxels(mri_chan_small, facecolors='#3eb19c20', shade=False) 
        ax.view_init(angle_view, angle)
        fig1 = plt.gcf()
        fig1.savefig(f'{out_dir}/{angle}.png')
#USes images created using create_images method to create a gif of brain spinning on vertical axis
def make_gif(in_dir,out_file,angle_num):
    images = []
    filenames = [f'{in_dir}/{x}.png' for x in range(0,360,angle_num)]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(out_file, images)

    
def make_ax(grid=False,fig_size = (50,25)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

#
def make_smaller(seg, factor = 4):
    smal = seg[0:(numpy.shape(seg)[0]):factor]
    smal =  numpy.stack([smal[x][0:(numpy.shape(seg)[1]):factor] for x in range((numpy.shape(smal)[0]))])
    return  numpy.stack([[smal[x][y][0:(numpy.shape(seg)[2]):factor] for x in range((numpy.shape(smal)[0]))] for y in range((numpy.shape(smal)[0]))]) 
