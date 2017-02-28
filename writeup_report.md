#**Behavioral Cloning** 

##Project Report

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/original.png "original histogram"
[image2]: ./examples/downsampled.png "downsampled histogram"
[image3]: ./examples/final_histogram.png "augmented data histogram"
[image4]: ./examples/error128.png "training/validation error during training epochs"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is similar to the one described in the Nvidia end-2-end paper. After cropping (65 pixles from the top, 25 from the bottom) and normalization (Keras Lambda layer), it applies 4 convolution layers with RELU activations. After that, it applies 4 fully connected layers with RELU's, and dropout of 0.5 to the first dense layer.

####2. Attempts to reduce overfitting in the model

The model contains a single dropout layer in order to reduce overfitting (train.py lines 158). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). Initial learning rate was 0.001. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only the sample data provided by Udacity. The images from the left and right cameras were used with an offset of +-0.25 to the steering angle.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to train and validate on different data, until finding a model with a 0.022 loss on the validation set.

I initially tried the network described in the Nvidia paper. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. After witnessing overfitting, I applied a dropout layer to the first fully-connected layer.

In parallel, I refined the training data. Details are in what follows.

The final step was to run the simulator to see how well the car was driving around track one. The car stayed on the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (train.py lines 132-166) has been described in the previous section.


####3. Creation of the Training Set & Training Process

My goal was to succeed in creating an agent without recording additional training data. Therefore, I only augmented the sample data provided by Udacity.

I started by making the data more uniform, decreasing the number examples with 0 steering angle. By doing this, the steering angles in the dataset turned from this:

![alt text][image1]

to this:

![alt text][image2]

When loading the data for training, I performed the following augmentations randomly:

* loading with equal probability an image from the center/right/left camera, offsetting the angle of side images with +-0.25.
* flipping the image and the steering angle with probability 0.5.
* applying random changes to the brightness and saturation of the image with probability 0.3.
* darkening a random square in the image, with probability 0.3. This was done to handle shadows better.

Eventually, the following is a histogram of angles that the model trains on (the generated data):

![alt text][image3]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the following plot of the training and validation error as a function of epochs:

![alt text][image4]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
