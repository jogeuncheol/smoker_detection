import tensorflow as tf
import cv2
import numpy as np

loaded_model = tf.keras.models.load_model('./model/save_model_3')
path = 'E:/workspace/python/BlazePose/data/validation_data/nonsmoking/260.jpg'
# 6. 모델 사용하기
img_height = 180
img_width = 180
test1 = cv2.imread(path)
test1 = cv2.resize(test1, dsize=(img_height, img_width))
test1 = test1.reshape(1, img_height, img_width, 3).astype('float32')# / 255.

# test2 = cv2.imread('data/smoke9.jpg')
path = 'E:/workspace/python/BlazePose/data/room4_e_smoke/room4_mp41395.jpg'
test2 = cv2.imread(path)
test2 = cv2.resize(test2, dsize=(img_height, img_width))
test2 = test2.reshape(1, img_height, img_width, 3).astype('float32')# / 255.

pred1 = loaded_model.predict(test1)
pred1 = tf.nn.softmax(pred1)
# pred1 = tf.where(pred1 < 0.5, 0, 1)
# pred1 = tf.keras.applications.mobilenet.decode_predictions(pred1)
pred2 = loaded_model.predict(test2)
pred2 = tf.nn.softmax(pred2)
# pred2 = tf.where(pred2 < 0.5, 0, 1)
# pred2 = tf.keras.applications.mobilenet.decode_predictions(pred2)
# game_pred = tf.keras.applications.mobilenet_v3.decode_predictions(game_pred, top=5)
# died_pred = tf.keras.applications.mobilenet_v3.decode_predictions(died_pred, top=5)
class_names = ['nonsmoking', 'smoking']
print("pred1", class_names[np.argmax(pred1)])
print("pred2", class_names[np.argmax(pred2)])
