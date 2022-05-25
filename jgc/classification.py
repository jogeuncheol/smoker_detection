import tensorflow as tf
import cv2
import numpy as np


class Model:
    def __init__(self):
        self.loaded_model = tf.keras.models.load_model('./save_model_3')
        self.class_names = ['nonsmoking', 'smoking']

    def image_classification(self, image):
        image = cv2.resize(image, dsize=(180, 180))
        image = image.reshape(1, 180, 180, 3).astype('float32')  # / 255.
        prediction = self.loaded_model.predict(image)
        prediction = tf.nn.softmax(prediction)
        class_name = self.class_names[np.argmax(prediction)]
        print("predict", class_name)
        if class_name == 'smoking':
            return True
        return False
