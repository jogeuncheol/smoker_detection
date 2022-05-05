import os
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import PIL
import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2

print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     print("Failed to limit GPU Memory")

train_path = "E:/workspace/학교_수업/4학년자료/22종프/smoking_dataset/7b52hhzs3r-1/smokingVSnotsmoking/dataset/training_data/" # 'E:\workspace\ML_study\my_train_set'
valid_path = "E:/workspace/학교_수업/4학년자료/22종프/smoking_dataset/7b52hhzs3r-1/smokingVSnotsmoking/dataset/validation_data/" # 'E:\workspace\ML_study\my_test_set'

# train_generator = ImageDataGenerator(rotation_range=15,
#                                      zoom_range=0.15,
#                                      width_shift_range=0.2,
#                                      height_shift_range=0.2,
#                                      shear_range=0.15,
#                                      horizontal_flip=True)
train_generator = image_dataset_from_directory(train_path,
                                    shuffle=True,
                                    image_size=(224, 224),
                                    batch_size=32)

# valid_generator = ImageDataGenerator(rotation_range=15,
#                                      zoom_range=0.15,
#                                      width_shift_range=0.2,
#                                      height_shift_range=0.2,
#                                      shear_range=0.15,
#                                      horizontal_flip=True)
valid_generator = image_dataset_from_directory(valid_path,
                                    shuffle=True,
                                    image_size=(224, 224),
                                    batch_size=32)

# check class image
class_names = train_generator.class_names
#
# plt.figure(figsize=(10, 10))
# for images, labels in train_generator.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# buffering prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_generator = train_generator.prefetch(buffer_size=AUTOTUNE)
valid_generator = valid_generator.prefetch(buffer_size=AUTOTUNE)

# data augmentation
data_aug = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# BUILD MODEL
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_batch, label_batch = next(iter(train_generator))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# convolution base model freeze
base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_aug(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
# x = Flatten(name="flatten")(x)
# x = Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = prediction_layer(x)
# outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = AveragePooling2D(pool_size=(7, 7))(x)
# x = Dense(512, activation='relu')(x)
# x = Dense(256, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# preds = Dense(2, activation='sigmoid')(x)
# goal::binary classification

# model = Model(inputs=base_model.input, outputs=preds)

# FREEZE PRETRAINED LAYERS
# for layers in model.layers[:-5]:
#     layers.trainable = False

# model.summary()
# plot_model(model, show_shapes=True, dpi=80)

# COMPILE AND FitMODEL
Batch_Size = 32
epochs = 50

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()
print(len(model.trainable_variables))

loss0, accuracy0 = model.evaluate(valid_generator)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# model fit :: training
history = model.fit(train_generator,
                    validation_data=valid_generator,
                    epochs=epochs)

# fine tuning
base_model.trainable = True

print("Number of layers in the base model : ", len(base_model.layers))
fine_tune_at = 150
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001/10),
              metrics=['accuracy'])
model.summary()
print(len(model.trainable_variables))

fine_tune_epochs = 50
total_epochs =  50 + fine_tune_epochs

history_fine = model.fit(train_generator,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=valid_generator)

model.save('./model/save_model')
model.save('./model/model', save_format='h5')
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("ploy.jpg")

N = total_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(epochs, N), history_fine.history["loss"], label="train_loss")
plt.plot(np.arange(epochs, N), history_fine.history["val_loss"], label="val_loss")
plt.plot(np.arange(epochs, N), history_fine.history["accuracy"], label="train_acc")
plt.plot(np.arange(epochs, N), history_fine.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("ploy_fine.jpg")

# 6. 모델 사용하기
test1 = cv2.imread('testing_data/abc001.jpg')
test1 = cv2.resize(test1, dsize=(224, 224))
test1 = test1.reshape(1, 224, 224, 3).astype('float32')# / 255.

test2 = cv2.imread('testing_data/abc301.jpg')
test2 = cv2.resize(test2, dsize=(224, 224))
test2 = test2.reshape(1, 224, 224, 3).astype('float32')# / 255.

pred1 = model.predict(test1)
pred1 = tf.nn.sigmoid(pred1)
pred1 = tf.where(pred1 < 0.5, 0, 1)
# pred1 = tf.keras.applications.mobilenet.decode_predictions(pred1)
pred2 = model.predict(test2)
pred2 = tf.nn.sigmoid(pred2)
pred2 = tf.where(pred2 < 0.5, 0, 1)
# pred2 = tf.keras.applications.mobilenet.decode_predictions(pred2)
# game_pred = tf.keras.applications.mobilenet_v3.decode_predictions(game_pred, top=5)
# died_pred = tf.keras.applications.mobilenet_v3.decode_predictions(died_pred, top=5)

print("pred1", pred1)
print("pred2", pred2)