# Implementation of the paper "Accelerating the Super-Resolution Convolutional Neural Network"

Dong, Chao, Chen Change Loy, and Xiaoou Tang. "Accelerating the super-resolution convolutional neural network." Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14. Springer International Publishing, 2016.


### Introduction:

Single image super-resolution (SR) aims at recovering a high-resolution (HR) image from a given low-resolution (LR) one. 
Super-Resolution Convolutional Neural Network (SRCNN) has drawn considerable attention due to its simple network structure and excellent restoration quality. 
Though SRCNN is already faster than most previous learning-based methods, the processing speed on large images is still unsatisfactory. 

Two inherent limitations that restrict its running speed. 
First, as a pre-processing step, the original LR image needs to be upsampled to the desired size using bicubic interpolation to form the input. Thus the computation complexity of SRCNN grows quadratically with the spatial size of the HR image
The second restriction lies on the costly non-linear mapping step. In SRCNN, input image patches are projected on a high-dimensional LR feature space, then followed by a complex mapping to another high-dimensional HR feature space. 


### FRSCNN:

To solve the first problem, the researchers adopt a deconvolution layer to replace the bicubic interpolation. To further ease the computational burden, they place the deconvolution layer at the end of the network, then the computational complexity is only proportional to the spatial size of the original LR image.

For the second problem, they add a shrinking and an expanding layer at the beginning and the end of the mapping layer separately to restrict mapping in a low-dimensional feature space. Furthermore, they decompose a single wide mapping layer into several layers with a fixed filter size 3 × 3

<img width="436" alt="image" src="https://user-images.githubusercontent.com/73118229/235309213-b43d01ce-a167-4088-bd2b-552e75b908f0.png">


### Model Architecture

1. Feature Extraction: FSRCNN performs feature extraction on the original LR image without interpolation; Conv layer = Conv(5, d, 1)

2. Shrinking: We add a shrinking layer after the feature extraction layer to reduce the LR feature dimension d; Conv layer = Conv(1, s, d)

3. Non linear Mapping: Conv layer = m × Conv(3, s, s)

4. Expanding: Conv layer = Conv(1, d, s)

5. Deconvolution: Deconv layer = DeConv(9, 1, d)

6. Activation function after every conv layer: PReLU

7. Cost function: Mean square error (MSE) 

8. Optimizer: Stochastic gradient descent. 

Final config :

Conv(5, d, 1)−P ReLU −Conv(1, s, d)−P ReLU −m×Conv(3, s, s)− P ReLU − Conv(1, d, s) − P ReLU − DeConv(9, 1, d)
Three sensitive variables (i.e., the LR feature dimension d, the number of shrinking filters s, and the mapping depth m) governing the performance and speed. For simplicity, we represent a FSRCNN network as F SRCNN(d, s, m).


### Training Details
1. Data -
 
DIV2K dataset. Has only HR images. Reshuffled it to create 700 : 100 : 100 for train : validation : test images.
NOTE: We used this dataset instead of the T91/ Gen100 suggested in the paper because this huge dataset was specifically developed for super resolution tasks.
Augmentations to the HR images and then bicubic downsampling to get LR images

2. Model Building and Training -
 
Layers: Conv(5, d, 1) −> P ReLU −> Conv(1, s, d) −> P ReLU −> m×Conv(3, s, s) −> P ReLU −> Conv(1, d, s) −> P ReLU −> DeConv(9, 1, d)
Padding: zero
Initializer: He Normal
Hyperparameters: d = 56, s = 12, m = 4, PRelu learnable parameter alpha = (shared_axes=[1, 2])
Learning rate: 0.001 (ReduceLROnPlateau callback to decrease the learning rate each time train loss stops falling.)
Epochs: 500 (maximum; We use EarlyStopping callback to finish training when validation loss stops falling)
Batch size: 30
Cost Function: Mean Squared Error (MSE)
Optimiser: Stochastic Gradient Descent (SGD) and RMSProp (Our optimisation)

3. Model Evaluation - 

Peak Signal to Noise Ratio (PSNR)
Visual evaluation : against bilinear upsampling

### Results
We can observe that our hypothesis of replacing SGD with RMSProp is successful (validation loss of 0.0266 in SGD v/s validation loss of 0.0078 in RMSProp)











