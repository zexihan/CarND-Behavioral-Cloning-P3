# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture.png "CNN Architecture"
[image2]: ./images/center.png "Center Camera"
[image3]: ./images/left.png "Left Camera"
[image4]: ./images/right.png "Right Camera"
[image5]: ./images/flipping.png "Flipped Image"
[image6]: ./images/cropping.png "Cropped Image"
[image7]: ./images/resizing.png "Resized Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the pipeline script to create and train the model
* network.py containing the network architecture
* transform.py for image transformation functions at preprocessing step
* drive.py for driving the car in autonomous mode
* saved_models/behavior_cloning_nvidia_model.003.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 is the result video recording the vehicle driving autonomously around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 53 (model.py lines 18-24) The model includes ELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Udacity sample data was used for training.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Nvidia architecture which has been proven to be very successful in self-driving car tasks.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with a 8:2 ratio. Since data augmentation technique was well-used, the mean squared error was low both on the training and validation steps.

To combat the overfitting and improve the model's generalization performance, I made an effort to do the data augmentation by cropping, flipping, and resizing.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track with my initial models which was only trained on the raw images by the center camera. To improve the driving behavior in these cases, I utilized all images from the three front cameras on the vehicle. During training, you want to feed the left and right camera images to your model as if they were coming from the center camera. This way, you can teach your model how to steer if the car drifts off to the left or the right.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture The code below shows the network architecture, which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. 

```python
model = Sequential([
    # Normalization Layer
    Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)),
    # Conv Layer 1
    Conv2D(filters = 24, 
            kernel_size = 5, 
            strides = 2, 
            activation="elu",
            kernel_regularizer=l2(0.001)),
    # Conv Layer 2
    Conv2D(filters = 36, 
            kernel_size = 5, 
            strides= 2, 
            activation="elu",
            kernel_regularizer=l2(0.001)),
    # Conv Layer 3
    Conv2D(filters = 48, 
            kernel_size = 5, 
            strides= 2, 
            activation="elu",,
            kernel_regularizer=l2(0.001)),
    # Conv Layer 4
    Conv2D(filters = 64, 
            kernel_size = 3, 
            strides= 1, 
            activation="elu",
            kernel_regularizer=l2(0.001)),
    # Conv Layer 5
    Conv2D(filters = 64, 
            kernel_size = 3, 
            strides= 1, 
            activation="elu",
            kernel_regularizer=l2(0.001)),
    # Flatten Layer
    Flatten(),
    # Fully-connected Layer 1
    Dense(units = 100, 
          activation="elu", 
          kernel_regularizer=l2(0.001)),
    # Fully-connected Layer 2
    Dense(units = 50, 
          activation="elu", 
          kernel_regularizer=l2(0.001)),
    # Fully-connected Layer 3
    Dense(units = 10, 
          activation="elu", 
          kernel_regularizer=l2(0.001)),
    # Output Layer
    Dense(units = 1, 
          activation="elu", 
          kernel_regularizer=l2(0.001)),
])
```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.

The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

We follow the five convolutional layers with three fully connected layers, leading to a final output control value which is the inverse-turning-radius. The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer if the car drifts off to the left or the right. 

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would help with the steering. For example, here is an image that has then been flipped, cropped, and resized:

![alt text][image5]
![alt text][image6]
![alt text][image7]

The given sample data set (minus the 20% validation set) has 24108 images when the data is augemented. The network was then trained for 5 epochs an GTX 1080 Ti GPU for approximately 4.8 hours.

```bash
Epoch 1/5
12856/12856 [==============================] - 3722s 289ms/step - loss: 0.0879 - val_loss: 0.10447/behavior_cloning_nvidia_model.001.h5
Epoch 2/5
12856/12856 [==============================] - 3586s 279ms/step - loss: 0.0305 - val_loss: 0.020103007/behavior_cloning_nvidia_model.002.h5
Epoch 3/5
12856/12856 [==============================] - 3579s 278ms/step - loss: 0.0433 - val_loss: 0.020003007/behavior_cloning_nvidia_model.003.h5
Epoch 4/5
12856/12856 [==============================] - 3564s 277ms/step - loss: 0.0611 - val_loss: 0.0309
Epoch 5/5
12856/12856 [==============================] - 3570s 278ms/step - loss: 0.0431 - val_loss: 0.0573
```