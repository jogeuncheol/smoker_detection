import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import PIL
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt

valid_path = 'E:/workspace/ML_study/my_test_set'
test_path = 'E:/workspace/학교_수업/4학년자료/22종프/smoking_dataset/7b52hhzs3r-1/smokingVSnotsmoking/dataset/testing_data/'
testing_dataset = image_dataset_from_directory(test_path,
                                               shuffle=True,
                                               batch_size=32,
                                               image_size=(224, 224))
class_names = testing_dataset.class_names

loaded_model = tf.keras.models.load_model('./model/save_model')
for i in loaded_model.get_config():
    print(i)

loss, accuracy = loaded_model.evaluate(testing_dataset)
print('Test accuracy : ', accuracy)

#Retrieve a batch of images from the test set
image_batch, label_batch = testing_dataset.as_numpy_iterator().next()
predictions = loaded_model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
plt.savefig("predict1.jpg")

# 6. 모델 사용하기
test1 = cv2.imread('data/non_smoke3.jpg')
test1 = cv2.resize(test1, dsize=(224, 224))
test1 = test1.reshape(1, 224, 224, 3).astype('float32')# / 255.

test2 = cv2.imread('data/smoke3.jpg')
test2 = cv2.resize(test2, dsize=(224, 224))
test2 = test2.reshape(1, 224, 224, 3).astype('float32')# / 255.

pred1 = loaded_model.predict(test1)
pred1 = tf.nn.sigmoid(pred1)
pred1 = tf.where(pred1 < 0.5, 0, 1)
# pred1 = tf.keras.applications.mobilenet.decode_predictions(pred1)
pred2 = loaded_model.predict(test2)
pred2 = tf.nn.sigmoid(pred2)
pred2 = tf.where(pred2 < 0.5, 0, 1)
# pred2 = tf.keras.applications.mobilenet.decode_predictions(pred2)
# game_pred = tf.keras.applications.mobilenet_v3.decode_predictions(game_pred, top=5)
# died_pred = tf.keras.applications.mobilenet_v3.decode_predictions(died_pred, top=5)

print("pred1", pred1)
print("pred2", pred2)