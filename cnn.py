from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from PIL import Image
import os, os.path
import numpy as np

def read_images(path):
    vidcap = cv2.VideoCapture(path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, a = vidcap.read()

    frames = []
    for i in range(1, length):
        if i % 30 == 0:
            a = b
        success, b = vidcap.read()
        c = cv2.subtract(b, a)
        frames.append(c[:, :, 0])
    return np.array(frames)

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_x, img_y = 640, 360

timelapse_frames = read_images("/input/timelapse.mp4")
print(timelapse_frames.shape)
video_frames = read_images("/input/video.mp4")

# load the MNIST data set, which already splits into train and test sets for us
x_train = np.concatenate((timelapse_frames[0:3000], video_frames[0:3000]))
y_train = np.concatenate((np.ones(3000), np.zeros(3000)))
x_test = np.concatenate((timelapse_frames[3000:3600], video_frames[3000:3600]))
y_test = np.concatenate((np.ones(3000), np.zeros(3000)))
print(x_train.shape)

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
