from helpers import *
from keras.layers import Dense
import keras.backend as K
from keras.applications import vgg16
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
import os
import warnings
warnings.simplefilter("ignore")
from keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')
root_dir = os.getcwd()
train_dir = (root_dir + '/datasets/training/')
valid_dir = (root_dir + '/datasets/validation/')

#Training data generator
training_dataset = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=40,
    zoom_range=0.1,
    rescale=1. / 255)
training_generator = training_dataset.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

# Validation data generator

validation_dataset = ImageDataGenerator(rescale=1. / 255)
validation_data_generator = validation_dataset.flow_from_directory(
    directory=valid_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=30,
    class_mode="categorical",
    shuffle=True,
    seed=20
)

##############################
# MODEL FROM SCRATCH
#############################
model = makefromsratch()

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
STEP_SIZE_TRAIN = training_generator.n // training_generator.batch_size
STEP_SIZE_VALID = validation_data_generator.n // validation_data_generator.batch_size

History = model.fit_generator(generator=training_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=validation_data_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=10,
                              verbose=1,
                              callbacks=[learning_rate_reduction]
                              )
curves_plot(History)


###################################################################
#       Transfer Learning
################################################################
vgg16_model = vgg16.VGG16(weights='imagenet')


pretrained_model = Sequential()

for layer in vgg16_model.layers[:-1]:
    pretrained_model.add(layer)
for layer in pretrained_model.layers:
    layer.trainable = False

pretrained_model.add(Dense(10, activation='softmax'))

pretrained_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

pretrained_history = pretrained_model.fit_generator(generator=training_generator,
                                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                                    validation_data=validation_data_generator,
                                                    validation_steps=STEP_SIZE_VALID,
                                                    epochs=10,
                                                    verbose=1,
                                                    callbacks=[learning_rate_reduction])
curves_plot(pretrained_history)

########################################################################
#               Console Log

########################################################################

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# batch_normalization (BatchNo (None, 224, 224, 3)       12
# _________________________________________________________________
# conv2d (Conv2D)              (None, 222, 222, 32)      896
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 111, 111, 32)      0
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 111, 111, 32)      128
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 109, 109, 64)      18496
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 54, 54, 64)        0
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 54, 54, 64)        256
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 52, 52, 128)       73856
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 26, 26, 128)       0
# _________________________________________________________________
# batch_normalization_3 (Batch (None, 26, 26, 128)       512
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 24, 24, 256)       295168
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 12, 12, 256)       0
# _________________________________________________________________
# batch_normalization_4 (Batch (None, 12, 12, 256)       1024
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 10, 10, 512)       1180160
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 5, 5, 512)         0
# _________________________________________________________________
# batch_normalization_5 (Batch (None, 5, 5, 512)         2048
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 512)               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                5130
# =================================================================
# Total params: 1,577,686
# Trainable params: 1,575,696
# Non-trainable params: 1,990
# _________________________________________________________________
# WARNING:tensorflow:From C:/pycharmprojects/cmsc828c_proj2/Part2_transferlearning:81: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
# Instructions for updating:
# Please use Model.fit, which supports generators.
# Epoch 1/10
# 34/34 [==============================] - 65s 2s/step - loss: 2.3272 - accuracy: 0.3386 - val_loss: 3.1621 - val_accuracy: 0.0963 - lr: 0.0010
# Epoch 2/10
# 34/34 [==============================] - 68s 2s/step - loss: 1.5255 - accuracy: 0.4634 - val_loss: 4.8846 - val_accuracy: 0.0963 - lr: 0.0010
# Epoch 3/10
# 34/34 [==============================] - 69s 2s/step - loss: 1.4040 - accuracy: 0.5178 - val_loss: 4.0111 - val_accuracy: 0.1704 - lr: 0.0010
# Epoch 4/10
# 34/34 [==============================] - 69s 2s/step - loss: 1.3085 - accuracy: 0.5385 - val_loss: 4.3464 - val_accuracy: 0.1259 - lr: 0.0010
# Epoch 5/10
# 34/34 [==============================] - 68s 2s/step - loss: 1.2010 - accuracy: 0.5854 - val_loss: 4.4103 - val_accuracy: 0.1741 - lr: 0.0010
# Epoch 6/10
# 34/34 [==============================] - 69s 2s/step - loss: 1.1757 - accuracy: 0.5994 - val_loss: 4.3404 - val_accuracy: 0.1185 - lr: 0.0010
# Epoch 7/10
# 34/34 [==============================] - ETA: 0s - loss: 1.0225 - accuracy: 0.6473
# Epoch 00007: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 34/34 [==============================] - 68s 2s/step - loss: 1.0225 - accuracy: 0.6473 - val_loss: 4.1969 - val_accuracy: 0.1741 - lr: 0.0010
# Epoch 8/10
# 34/34 [==============================] - 70s 2s/step - loss: 0.8704 - accuracy: 0.6876 - val_loss: 3.6838 - val_accuracy: 0.1889 - lr: 5.0000e-04
# Epoch 9/10
# 34/34 [==============================] - 69s 2s/step - loss: 0.8126 - accuracy: 0.6998 - val_loss: 2.8301 - val_accuracy: 0.2333 - lr: 5.0000e-04
# Epoch 10/10
# 34/34 [==============================] - 69s 2s/step - loss: 0.7576 - accuracy: 0.7383 - val_loss: 2.3515 - val_accuracy: 0.3519 - lr: 5.0000e-04
# Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
# 553467904/553467096 [==============================] - 17s 0us/step
# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________
# Epoch 1/10
# 34/34 [==============================] - 190s 6s/step - loss: 1.4621 - accuracy: 0.1717 - val_loss: 1.9746 - val_accuracy: 0.3407 - lr: 0.0010
# Epoch 2/10
# 34/34 [==============================] - 192s 6s/step - loss: 0.9353 - accuracy: 0.3218 - val_loss: 1.7938 - val_accuracy: 0.3630 - lr: 0.0010
# Epoch 3/10
# 34/34 [==============================] - 192s 6s/step - loss: 0.7885 - accuracy: 0.5630 - val_loss: 1.6328 - val_accuracy: 0.4185 - lr: 0.0010
# Epoch 4/10
# 34/34 [==============================] - 192s 6s/step - loss: 0.5594 - accuracy: 0.6728 - val_loss: 1.4458 - val_accuracy: 0.5037 - lr: 0.0010
# Epoch 5/10
# 34/34 [==============================] - 192s 6s/step - loss: 0.5484 - accuracy: 0.7606 - val_loss: 1.3900 - val_accuracy: 0.5519 - lr: 0.0010
# Epoch 6/10
# 34/34 [==============================] - 195s 6s/step - loss: 0.4212 - accuracy: 0.8954 - val_loss: 1.3402 - val_accuracy: 0.5519 - lr: 0.0010
# Epoch 7/10
# 34/34 [==============================] - ETA: 0s - loss: 0.4126 - accuracy: 0.5159
# Epoch 00007: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 34/34 [==============================] - 191s 6s/step - loss: 0.4126 - accuracy: 0.9159 - val_loss: 1.3663 - val_accuracy: 0.5111 - lr: 0.0010
# Epoch 8/10
# 34/34 [==============================] - 207s 6s/step - loss: 0.2701 - accuracy: 0.9807 - val_loss: 1.2020 - val_accuracy: 0.6148 - lr: 5.0000e-04
# Epoch 9/10
# 34/34 [==============================] - 194s 6s/step - loss: 0.2763 - accuracy: 0.9750 - val_loss: 1.1833 - val_accuracy: 0.5852 - lr: 5.0000e-04
# Epoch 10/10
# 34/34 [==============================] - ETA: 0s - loss: 1.2544 - accuracy: 0.5947
# Epoch 00010: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
# 34/34 [==============================] - 194s 6s/step - loss: 1.2544 - accuracy: 0.9747 - val_loss: 1.1900 - val_accuracy: 0.5926 - lr: 5.0000e-04
