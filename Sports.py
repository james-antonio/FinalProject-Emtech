import streamlit as st
import tensorflow as tf
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('sports.hdf5')
  return model
model=load_model()
st.write("""
# Sport Celebrity Image Classifier"""
)

image = Image.open('SPORTS.jpg')
st.image(image, caption='Sports Celebrity Classifier')

file=st.file_uploader("Choose photo from computer, either Kane Williamson, Kobe Bryant, Maria Sharapova, or Ronaldo",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Lion','Cheetah']
    string="The picture shown above is a : "+class_names[np.argmax(prediction)]
    st.success(string)
