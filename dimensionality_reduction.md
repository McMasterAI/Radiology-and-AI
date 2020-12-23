## Dealing with massive image files

### If a single image can fit into GPU memory
- Use distributed processing to load 1 image on each GPU, use multiple GPUs (at least, TensorFlow supports this). [link](https://www.tensorflow.org/guide/distributed_training)
- Fit an autoencoder and train using the internal representation.
   - Potentially interesting if a single image modality fits, but not all 4 at once
   - I tried this before and it didn't take that long even with batch size=1
- Use early strided convolution layers to reduce dimensionality. Used in U-net. [link](https://arxiv.org/abs/1505.04597)
- Image fusion
   - principal component analysis (this also works for image compression if you do it differently)
   - frequency-domain image fusion such as various shearlet transforms (I don't understand these, but here's a paper [link](https://journals.sagepub.com/doi/full/10.1177/1748301817741001))
   - I guess you could probably also use an autoencoder for this
   - This should reduce our 4-channel (4 neuroimaging types) image to have less channels containing the same information

### Works even if a single image can't fit into GPU memory
- Cropping
    - This probably works better if the images are registered to approximately the same space
- Slicing [Cameron's review with some of these](https://www.sciencedirect.com/science/article/pii/S187705091632587X)
    - Use 2-dimensional slices of 3D image, which each definitely fit in memory
    - (probably) can train models for each modality separately and average/use a less-GPU intensive model to combine them?
    - (probably) split image into smaller 3D patches for segmentation
- Downsampling: [this paper](https://nvlpubs.nist.gov/nistpubs/ir/2013/NIST.IR.7839.pdf) is not about neuroimaging at all but maybe has some insights?
    - Spectral truncation
        - Compute fast Fourier transform, reduce sampling rate, compute inverse FFT
        - I'm going to add wavelet transform here for similar reasons
    - Average pooling (take the average of 2x2x2 voxels)
    - Max pooling (take the maximum of 2x2x2 voxels)
    - Decimation/Gaussian blur with decimation (take every other line)
- Use a convolutional neural network that works on spectrally compressed images [link](https://www.sciencedirect.com/science/article/abs/pii/S0925231219310148)
    - probably really stupid
    - compute FFT, discrete cosine transform, or whatever
    - clip the spectrum to get rid of irrelevant high frequency noise
    - use a spectral convolutional neural network to compute everything in frequency domain
    - transform back to image domain 
   