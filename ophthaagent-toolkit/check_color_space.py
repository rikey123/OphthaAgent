import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "<ANON_ABS_PATH>"
IMG_PATH = "<ANON_ABS_PATH> Segmentation/1. Original Images/a. Training Set/IDRiD_01.jpg"

model = load_model(MODEL_PATH, compile=False)

img_bgr = cv2.imread(IMG_PATH)
img_bgr_resized = cv2.resize(img_bgr, (64, 64))
img_bgr_input = np.reshape(img_bgr_resized, [1, 64, 64, 3])

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb_resized = cv2.resize(img_rgb, (64, 64))
img_rgb_input = np.reshape(img_rgb_resized, [1, 64, 64, 3])

pred_bgr = model.predict(img_bgr_input)
pred_rgb = model.predict(img_rgb_input)
pred_bgr_norm = model.predict(img_bgr_input / 255.0)
pred_rgb_norm = model.predict(img_rgb_input / 255.0)

print(f"BGR Output: {pred_bgr[0][0]}")
print(f"RGB Output: {pred_rgb[0][0]}")
print(f"BGR Norm Output: {pred_bgr_norm[0][0]}")
print(f"RGB Norm Output: {pred_rgb_norm[0][0]}")
