# **Behavioral Cloning** 
Use Convolutional Neural Network to clone driving behavior

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* readme.md explains the structure of your network and training approach
* images support images for readme
* run1.mp4 result of code

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



## Model Architecture

I created my model based on [NVIDIA model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. Converse to the Nvidia model, input image was split to HSV planes before been passed to the network.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

The model contains 2 dropout layers in order to reduce overfitting. 

The model was trained and validated on separated data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer and the learning rate was set to 0.0001.  
   
 ![alt text](https://github.com/inokatsu/CarND-Behavior-Cloning/blob/master/images/nvidia_network.png, "NVIDIA network")        
Fig1 : NVIDIA network

The model looks like follows:
    
* Image normalization
* Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Drop out (0.6)
* Fully connected: neurons: 100, activation: ELU
* Drop out (0.6)
* Fully connected: neurons: 50, activation: ELU
* Fully connected: neurons: 10, activation: ELU
* Fully connected: neurons: 1 (output)



The below is an model structure output from Keras.

|Layer (type)| Output Shape | Param  |   Connected to |  
|:----:|:----:|:----:|:----:| 
|lambda_1(Lambda)|  (None, 66, 200, 3) |   0   |  lambda_input_1[0][0]  |         
|convolution2d_1(Convolution2D) | (None, 31, 98, 24) |   1824  |     lambda_1[0][0] |   
|convolution2d_2 (Convolution2D) | (None, 14, 47, 36)  |  21636   |    convolution2d_1[0][0] |     
|convolution2d_3 (Convolution2D) | (None, 5, 22, 48)  |   43248  |     convolution2d_2[0][0] |     
|convolution2d_4 (Convolution2D) | (None, 3, 20, 64)    | 27712     |  convolution2d_3[0][0] |     
|convolution2d_5 (Convolution2D) | (None, 1, 18, 64)   |  36928      | convolution2d_4[0][0]    | 
| flatten_1 (Flatten)             | (None, 1152)     |     0         |  convolution2d_5[0][0] |    
| dropout_1 (Dropout)        |      (None, 1152)     |     0         |  flatten_1[0][0] |          
|dense_1 (Dense)              |    (None, 100)         |  115300        | dropout_1[0][0]      |   
|dropout_1 (Dropout)       |       (None, 1152)    |      0         |  flatten_1[0][0]   |         | dense_2 (Dense)        |          (None, 50)     |       5050    |    dense_1[0][0]  |           
| dense_3 (Dense)        |          (None, 10)     |      510      |   dense_2[0][0]  |           
| dense_4 (Dense)        |          (None, 1)     |        11       |   dense_3[0][0] |            

Total params: 252,219  
Trainable params: 252,219  
Non-trainable params: 0  



## Training Process

### Data preprocessing

Udacity provided us the sample data which cointains 8037 images. The steering data of Udacity provided have mainly distributed on the small steering angle and it will cause poor performance. Therefore, I cut off the 99% of data where the value of steering is 0.0 angle and 80% of the data where the absolute value of steering within 0.1.


![alt text](https://github.com/inokatsu/CarND-Behavior-Cloning/blob/master/images/hist_sample.png)  
Fig2 : The original data distribution


![alt text](https://github.com/inokatsu/CarND-Behavior-Cloning/blob/master/images/hist_sample_cut.png) 
Fig3 : The steering data distribution after cutting off

Then, following preprocessing has done.

* crop image : cut off the sky and the car front parts
* resize image : 160x320 to 66x200 for NVIDIA model
* blur image : smoothing image to remove noises
* change colur space : convert from RGB to YUV for NVIDIA model

### Data augmentation

To augment the data set, I used following things:

* select image from right, center and left image randomly
* flip image left/right randomly
* change brightness randomly

If the left/right images are used, steering angle is adjusted by +/-0.2



I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 which is set experimentally. I tried the large number of epochs and it helped to lowering the loss, but it didn't lead to a good performance.

## Result
The result of this project is [here](https://youtu.be/yQKU0eNSMng).  
[![alt text](https://github.com/inokatsu/CarND-Behavior-Cloning/blob/master/images/Track_1_screenshot.png)](https://youtu.be/yQKU0eNSMng)

