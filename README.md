# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[recover1]: ./examples/recover1.jpg "Recovery Image"
[recover2]: ./examples/recover2.jpg "Recovery Image"
[recover3]: ./examples/recover3.jpg "Recovery Image"
[normal]: ./examples/normal.jpg "Normal Image"
[video1]: ./run2.mp4 "Video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network as described in the project overview lectures (model.py lines 69-81).

The data is normalized in the model using a Keras lambda layer (code line 70) followed by 5 convolutional layers. First three have a filter size of 5x5 with stride of 2. The next two convolution layers have a filter size of 3x3. Lastly the network has 4 fully connected layers.

RELU is used as activation in the convolution layers. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets (split from the data captured using the simulator) to ensure that the model was not overfitting (code lines 19, 60-61, 84). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

During the project overview lectures I was introduced to the Convolution Network used by NVIDIA for autonomous driving. I implemented the same network in Keras to see how it performs.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation set (20%). 

To combat the overfitting, I set the EPOCHS to 5, that was where both training and validation loss kept decreasing.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle was touching the side lane markers, so I added more laps of training data which included recovery training too.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-81) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x80x3 Image   							| 
| Lambda         		| Normalize the image  							| 
| Convolution 5x5     	| 2x2 stride, RELU activation 	|
| Convolution 5x5     	| 2x2 stride, RELU activation 	|
| Convolution 5x5     	| 2x2 stride, RELU activation 	|
| Convolution 3x3     	| 1x1 stride, RELU activation 	|
| Convolution 3x3     	| 1x1 stride, RELU activation 	|
| Flatten               |                               |
| Fully connected		| outputs = 100    				|
| Fully connected		| outputs = 50  				|
| Fully connected		| outputs = 10                  |
| Fully connected		| outputs = 1                  |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][normal]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return back to the center of lane. These images show what a recovery looks like starting from Facing left to coming parallel to the side lane marker and then returning to center:

![Left][recover1]
![Parallel][recover2]
![Center][recover3]


After the collection process, I had 26,904 frames and angles (data points). I then preprocessed this data by cropping the frames to focus only on the road which minimizes the effect of environment (sky, clouds) which is not instrumental in determining the driving direction.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as going beyond that the validation loss started going up and down. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Video of automated driving
[Automated Driving](https://github.com/psharm8/CarND-Behavioral-Cloning-P3/blob/master/run2.mp4)
