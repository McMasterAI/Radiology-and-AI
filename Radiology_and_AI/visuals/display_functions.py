from mpl_toolkits.mplot3d import Axes3D
import numpy
import imageio
import matplotlib.pyplot as plt

def display_brain_and_segs(seg_1,seg_2 = None,mri_chan=None,downsize_factor=5,fig_size=(50,25)):    
    ax = make_ax(True,fig_size=fig_size)
    seg_1_small = make_smaller(seg_1,downsize_factor)
    ax.voxels(seg_1_small, edgecolors='gray', shade=False)
    if type(seg_2) == type(None):
        seg_2_small = make_smaller(seg_2,downsize_factor)
        ax.voxels(seg_2_small, facecolors='#1f77b430', shade=False)
    if type(mri_chan) == type(None):
        mri_chan_small = make_smaller(seg_2,downsize_factor) 
        ax.voxels(mri_chan_small, facecolors='#3eb19c20', shade=False)                   
    plt.show() 

def gen_gif(out_dir,seg_1,seg_2 = None,mri_chan=None,downsize_factor=5,angle_num = 20,fig_size=(50,25)): 
    for angle in range(0, 360, angle_num):
        ax = make_ax(True,fig_size=(50,25))
        seg_1_small = make_smaller(seg_1,downsize_factor)
        ax.voxels(seg_1_small, edgecolors='gray', shade=False)
        if seg_2 != None:
            seg_2_small = make_smaller(seg_2,downsize_factor)
            ax.voxels(seg_2_small, facecolors='#1f77b430', shade=False)
        if mri_chan != None:
            mri_chan_small = make_smaller(seg_2,downsize_factor) 
            ax.voxels(mri_chan_small, facecolors='#3eb19c20', shade=False)   
        ax.view_init(30, angle)
        fig1 = plt.gcf()
        fig1.savefig(f'{outdir}/{angle}.png')


def make_gif(in_dir,out_file,angle_num):
    images = []
    filenames = [f'{indir}/{x}.png' for x in range(0,360,angle_num)]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(outfile, images)


def make_ax(grid=False,fig_size = (50,25)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

def make_smaller(seg, factor = 4):
  smal = seg[0:(numpy.shape(seg)[0]):factor]
  smal =  numpy.stack([smal[x][0:(numpy.shape(seg)[1]):factor] for x in range((numpy.shape(smal)[0]))])
  return  numpy.stack([[smal[x][y][0:(numpy.shape(seg)[2]):factor] for x in range((numpy.shape(smal)[0]))] for y in range((numpy.shape(smal)[0]))])    

