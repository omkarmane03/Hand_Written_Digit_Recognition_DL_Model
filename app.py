import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import cv2

st.header('Hand Written Digit Recognition Model')
model = load_model('C:\Python\Hand_Written_Digit_Recognition\Hand_Written_Digit_Recognition_Model.keras')
img =st.text_input('Enter Image name')

image = cv2.imread(img)[:,:,0]
image = np.invert(np.array([image]))
 
output = model.predict(image)
st.image(img, width=300)
stn = 'Digit in image is ' + str(np.argmax(output))
st.markdown(stn)