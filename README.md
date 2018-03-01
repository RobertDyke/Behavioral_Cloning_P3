# Behavioral_Cloning_P3
Trains a car on a simulator to drive around a test track without crashing.

  Drive.py records the control inputs while driving on the simulator. This can be used with a mouse and keyboard, but it works far better with a joystick. The joystick allows for smoother turns which is important for training the model. The simiuator is available at Udacity as the Term One simulator. Drive.py communicates with it via port 4567.
  The training code is in model.py.ipynb.

  The main variables you might want to adjust are batch_size, test_size, and epochs. The track is a loop. So that will produce a bias of turns in one direction during training. To counter this two lines of code are added.
  augmented_images.append(cv2.flip(image,1))
  augmented_measurements.append(measurement*(-1.0))
  Run half the time with the two lines and half the time with them commented out. (It just flips the image from left to right. Making the track look like it goes the opposite direction.)
  
  When the code runs three warnings will come up. Ignore these. They are just saying that Conv2D is being replaced in an upcoming version of Keras.
  The output of the code is a a keras model 'model.h5'.
  
  Below is a more detailed description of design choices. It is also availible with images in writeup_report.ipynb.
  
  

    Model Architecture and Training Strategy:

    My first attempt at the project used the CNN built during the lesson. It turned out to be insufficient. Subsequent attempts centered around collecting more driving data. However, greater amounts of data did not improve the performance. After a couple of days I slapped my forehead and cried, "D'oh!" I had been writing the new data to a different location. Thus I was getting no improvement as I was running the same old data set each time. In the mean time I had decided that the problem lay with the CNN. So, I changed mine to be the one in the Nvidia paper. I have no idea if the old CNN would have worked. To minimize the compute cycles needed I cropped the images by 20 pixels on each side, 20 on the bottom (removing the hood of the car), and 75 on the top (removing the sky and trees). I used three convolution layers of depths 16, 32, and 64. Each with 3X3 filters. After each convolution I used a max pooled layer. I flattened the output the used reducing Dense layers of 400, 100, 20, and 1. To reduce nonlinearity I used activated RELU on each convolution and each Dense except the last one. ADDENDUM: I wrote the original writeup in the middle of the night after many hours of coding. Hence, I forgot some details of the development of the CNN model. Much of the development of this project was spent on a bug hunt. The bug ended up being that I was always looking at the original data set, and not each new data set. "No matter what I do the model never improves!" Well, "D'oh! How could it? The original data set was insufficient, and never changed. While trying to figure this out I modified that model many times. Hence the deviation from the Nvidia CNN model. One of the papers I looked at used max-pooling between each convolution layer. This was one of the last changes I made. (Referenced paper: "Max-Pooling Convolution Neural Networks for Vision-based Hand Gesture Recognition" by Jawad Nai, et al.)

    Attempts to Reduce Overfitting in the Model.

    I trained the model on many different data sets. As new data sets were added I judged the effects on the simulator.Discussions on forums strongly suggested that a minimum of 100,000 images were needed. I drove the car around the track several more times in both directions, plus added several loops of 'recovery driving'. Also, all images and steering measurements were reversed to double the number of images. ADDENDUM: As part of my second submission I added a dropout layer after the first convolution. This made the outcome much worse. I then moved the dropout layer to after the last convolution. This worked. My best guess as to why is that the model needs to digest all of the data for the first few steps in order to get the model on "the right track". After that data can be removed with no ill affect.

    Model Parameter Tuning

    The model used an Adam Optimizer , so the learning rate did not have to be adjusted manually.

    Appropriate Training Data

    I used only images from the center camera, and augemented images derived from the center camera.

Training Strategy

    Solution Design Approach

    My original attempt was to use the model built during the lesson. This proven insufficient. Next I used the architecture in the Nvidia paper. In the end it worked. As I modified the design I would run simulations to judge the effects. Initially I tried using just Model.fit. This caused memory failures once my data sets became large enough. It took several days to get Model.fit_generator to work.

    ADDENDUM: In an effort to improve computing time I cropped less useful parts of the images out. After several trials I settled on removing the top row of 75 pixels, the bottom 25, and 20 from the left and right sides. A sample original and cropped version are shown below.

    The histogram below shows an even distribution of steering angles. If all test loops had been in one direction there would have been an imbalence of turns to one direction. By including some data gathering loops going in the opposite direction and augmenting the data with mirror images I have producted a balenced data set.

On each run I chose 10 Epochs. As the last three cases that I ran show the "sweet spot" moves around. In the first case it appears to be after Epoch 9, 2nd case after epoch 9, and 3rd after epoch 7. This averages out to epoch 8 being the sweet spot. So a better model could be produced by only doing 8 epochs.

Final model from 1st submission

Epoch 1/10 131/130 [==============================] - 715s - loss: 0.0124 - val_loss: 0.0081 Epoch 2/10 131/130 [==============================] - 691s - loss: 0.0070 - val_loss: 0.0078 Epoch 3/10 131/130 [==============================] - 714s - loss: 0.0064 - val_loss: 0.0073 Epoch 4/10 131/130 [==============================] - 709s - loss: 0.0055 - val_loss: 0.0069 Epoch 5/10 131/130 [==============================] - 687s - loss: 0.0050 - val_loss: 0.0067 Epoch 6/10 131/130 [==============================] - 688s - loss: 0.0042 - val_loss: 0.0067 Epoch 7/10 131/130 [==============================] - 685s - loss: 0.0038 - val_loss: 0.0065 Epoch 8/10 131/130 [==============================] - 1774s - loss: 0.0030 - val_loss: 0.0065 Epoch 9/10 131/130 [==============================] - 687s - loss: 0.0026 - val_loss: 0.0058 Epoch 10/10 131/130 [==============================] - 686s - loss: 0.0022 - val_loss: 0.0065 dict_keys(['loss', 'val_loss'])

April 1 2017 After addition of drop out layer after 1st convolution layer

Epoch 1/10 131/130 [==============================] - 737s - loss: 0.0134 - val_loss: 0.0095 Epoch 2/10 131/130 [==============================] - 721s - loss: 0.0076 - val_loss: 0.0078 Epoch 3/10 131/130 [==============================] - 720s - loss: 0.0071 - val_loss: 0.0079 Epoch 4/10 131/130 [==============================] - 719s - loss: 0.0065 - val_loss: 0.0076 Epoch 5/10 131/130 [==============================] - 718s - loss: 0.0062 - val_loss: 0.0070 Epoch 6/10 131/130 [==============================] - 723s - loss: 0.0057 - val_loss: 0.0078 Epoch 7/10 131/130 [==============================] - 735s - loss: 0.0052 - val_loss: 0.0066 Epoch 8/10 131/130 [==============================] - 754s - loss: 0.0048 - val_loss: 0.0066 Epoch 9/10 131/130 [==============================] - 722s - loss: 0.0043 - val_loss: 0.0065 Epoch 10/10 131/130 [==============================] - 731s - loss: 0.0037 - val_loss: 0.0066 dict_keys(['val_loss', 'loss'])

7:46pm after moving dropout layer to end of convolutions. This is the final model for 2nd submission

Epoch 1/10 131/130 [==============================] - 703s - loss: 0.0125 - val_loss: 0.0074 Epoch 2/10 131/130 [==============================] - 700s - loss: 0.0077 - val_loss: 0.0077 Epoch 3/10 131/130 [==============================] - 700s - loss: 0.0071 - val_loss: 0.0065 Epoch 4/10 131/130 [==============================] - 700s - loss: 0.0064 - val_loss: 0.0062 Epoch 5/10 131/130 [==============================] - 699s - loss: 0.0058 - val_loss: 0.0062 Epoch 6/10 131/130 [==============================] - 701s - loss: 0.0054 - val_loss: 0.0064 Epoch 7/10 131/130 [==============================] - 701s - loss: 0.0049 - val_loss: 0.0058 Epoch 8/10 131/130 [==============================] - 726s - loss: 0.0044 - val_loss: 0.0059 Epoch 9/10 131/130 [==============================] - 733s - loss: 0.0038 - val_loss: 0.0059 Epoch 10/10 131/130 [==============================] - 741s - loss: 0.0034 - val_loss: 0.0062 dict_keys(['loss', 'val_loss'])

1.5. Model.fit_generator

Keras is currently in a awkward position as it migrates from Keras 1.0 to Keras 2.0. I found that fit_generator did not perform as advertised. The problem lies in the batch size. The old argument "samples_per_epoch" and "nb_val_samples" were designed to take the Number of Iterations per Batch. In Keras 2.0 these arguments are replaced with "steps_per_epoch" and "validation_steps". These want the Number of Batches per Epoch as their value. Supposedly you can still use the old arguments. You can, but they are now just the two new arguments renamed. Ergo, you must provide the Number of Batches per Epoch instead of the Number of Iterations per Batch. This is the problem that caused so many people to have fit_generator running ridiculously slow.

2.Final Model Architecture

To minimize the compute cycles needed I cropped the images by 20 pixels on each side, 20 on the bottom (removing the hood of the car), and 75 on the top (removing the sky and trees).I used three convolution layers of depths 16, 32, and 64. Each with 3X3 filters. After each convolution I used a max pooled layer. I flattened the output the used reducing Dense layers of 400, 100, 20, and 1. To reduce nonlinearity I used activated RELU on each convolution and each Dense except the last one.

    Creating the Training Set I recorded at least five loops around the track in each direction plus an augmented set of mirror images of that set. I used only the center camera. I suspected that the side camera's were unnecessary. After all, humans can drive with only a center view. I also had at least one loop of 'recovery' data. I would drive too far to the side then record myself driving back to center.

In Conclusion.

This was another awesome project! It took far longer than I expected, but that is programming. My main suggestion is to alter the lesson notes to reflect the problem with fit_generator. This should save students considerable time.
