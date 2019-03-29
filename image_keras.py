# DETECTING PRSENCE OF PENUMONIA FROM X-RAY IMAGES
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import pandas as pd
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import load_model

# IMPORTING DATASET
# creating links to the relevant folders 
root_dir = "C:/Users/User/Desktop/Software Projects/NUS Intership 070119/Deep Learning/X-Ray classification/chest_xray"
train_dir = root_dir + '/train'
val_dir = root_dir + '/val'
test_dir = root_dir + '/test'

nb_train = 5216
nb_val = 16
nb_test = 624
batch_size = 64

# DATA AUGMENTATION
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = batch_size, target_size = (64, 64), class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory(val_dir, batch_size = batch_size, target_size = (64, 64), class_mode = 'binary')
test_generator = test_datagen.flow_from_directory(test_dir, batch_size = batch_size, target_size = (64, 64), class_mode = 'binary')

# MODEL INITIALISATION
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (64, 64, 3)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
# adding a classifer on top of the convnet
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()

# COMPILING MODEL
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

# FITTING MODEL
nb_epochs = 50
train_sample_size = nb_train / batch_size
val_sample_size = nb_val / batch_size
test_sample_size = nb_test / batch_size

print(train_sample_size, val_sample_size, test_sample_size)

current_acc = 0
train_acc = []
train_loss = []
val_acc = []
val_loss =[]

for i in range(nb_epochs):
	if i > 0:
		model = load_model('my_model.h5')

	history = model.fit_generator(train_generator, steps_per_epoch = train_sample_size, epochs = 1,
								 validation_data = test_generator, validation_steps = test_sample_size)

	# appending the final training and validation accuracy and loss per epoch
	train_acc.append(history.history['acc'][-1])
	train_loss.append(history.history['loss'][-1])
	val_acc.append(history.history['val_acc'][-1])
	val_loss.append(history.history['val_loss'][-1])

	epoch_acc = history.history['val_acc'][-1]
	print('Validation acc:', epoch_acc)

	# next epoch will rerun the same model
	model.save('my_model.h5')

	if epoch_acc > current_acc:
		current_acc = epoch_acc
		print("Saving better model")
		model.save('best_model.h5')

epoch_index = range(1, nb_epochs + 1)

# GRAPHING OUT ACCURACY AND LOSS OF THE MODEL 
plt.plot(epoch_index, train_acc, 'bo', label = 'Training Accuracy')
plt.plot(epoch_index, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epoch_index, train_loss, 'bo', label = 'Training Loss')
plt.plot(epoch_index, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()