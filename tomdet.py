import matplotlib.pyplot as plt
import numpy as np
import cv2

from keras.preprocessing.image import ImageDataGenerator

tomato1 = cv2.imread('datasets/Train/Tomato/1.jpg')
tomato1 = cv2.cvtColor(tomato1,cv2.COLOR_BGR2RGB)
##plt.imshow(tomato1)
##plt.show()
##print(tomato1.shape)


Apple1 = cv2.imread('datasets/Train/Apple/1.jpg')
Apple1 = cv2.cvtColor(Apple1,cv2.COLOR_BGR2RGB)
##plt.imshow(tomato1)
##plt.show()
##print(Apple1.shape)

input_shape = (300,300,3)

image_gen = ImageDataGenerator(rotation_range=0,
                               width_shift_range=0.0,
                               height_shift_range=0.0,
                               rescale=1/255,
                               shear_range=0.0,
                               zoom_range=0.0,
                               horizontal_flip=True,
                               fill_mode='nearest'
                               )

##plt.imshow(image_gen.random_transform(tomato1))
##plt.show()



from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()

model.add(Conv2D(filters =32,kernel_size=(3,3),input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters =32,kernel_size=(3,3),input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters =32,kernel_size=(3,3),input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

##model.summary()

batch_size = 16

train_image_gen = image_gen.flow_from_directory('datasets/Train',
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory('datasets/Test',
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

##from keras.utils.np_utils import to_categorical
##
##test_image_gen = to_categorical(test_image_gen,3)
##train_image_gen = to_categorical(train_image_gen,3)

print(test_image_gen.class_indices)

results = model.fit_generator(train_image_gen,epochs=50,steps_per_epoch=100,
                              validation_data=test_image_gen,validation_steps=12)



print(results.history['acc'])


model.save('tom50model.h5')
































    
