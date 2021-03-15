from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as k
from helpers import *

(X_train, y_train), (X_test, y_test) = mnist.load_data()  # everytime loading data won't be so easy :)

#  dataset preparation
img_rows, img_cols = 28, 28

if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
num_category = 10
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)
y_train[0]

#model generation
mnist_model = mnist_cnnmodel(input_shape)

#compile model
mnist_model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
batch_size = 32
num_epoch = 10

model_history = mnist_model.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=num_epoch,
                      verbose=1,
                      validation_data=(X_test, y_test))
curves_plot(model_history)

########################################################################
#               Console Log

########################################################################
# Epoch 1/10
# 1875/1875 [==============================] - 55s 29ms/step - loss: 2.2379 - accuracy: 0.2463 - val_loss: 2.1411 - val_accuracy: 0.5925
# Epoch 2/10
# 1875/1875 [==============================] - 61s 33ms/step - loss: 2.0471 - accuracy: 0.4679 - val_loss: 1.8685 - val_accuracy: 0.6776
# Epoch 3/10
# 1875/1875 [==============================] - 61s 32ms/step - loss: 1.7440 - accuracy: 0.5703 - val_loss: 1.4688 - val_accuracy: 0.7319
# Epoch 4/10
# 1875/1875 [==============================] - 63s 33ms/step - loss: 1.4025 - accuracy: 0.6290 - val_loss: 1.0971 - val_accuracy: 0.7915
# Epoch 5/10
# 1875/1875 [==============================] - 61s 32ms/step - loss: 1.1431 - accuracy: 0.6743 - val_loss: 0.8501 - val_accuracy: 0.8248
# Epoch 6/10
# 1875/1875 [==============================] - 61s 32ms/step - loss: 0.9739 - accuracy: 0.7128 - val_loss: 0.6991 - val_accuracy: 0.8414
# Epoch 7/10
# 1875/1875 [==============================] - 58s 31ms/step - loss: 0.8621 - accuracy: 0.7386 - val_loss: 0.6026 - val_accuracy: 0.8547
# Epoch 8/10
# 1875/1875 [==============================] - 58s 31ms/step - loss: 0.7816 - accuracy: 0.7610 - val_loss: 0.5376 - val_accuracy: 0.8644
# Epoch 9/10
# 1875/1875 [==============================] - 61s 32ms/step - loss: 0.7260 - accuracy: 0.7762 - val_loss: 0.4927 - val_accuracy: 0.8711
# Epoch 10/10
# 1875/1875 [==============================] - 60s 32ms/step - loss: 0.6797 - accuracy: 0.7911 - val_loss: 0.4588 - val_accuracy: 0.8786
