import streamlit as st
import tensorflow as tf
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('sports2.hdf5')
  return model
model=load_model()
st.write("""
# Sports Celebrity Image Classifier"""
)

image = Image.open('SPORTS.jpg')
st.image(image, caption='Sports Celebrity Classifier')

with st.container():
    col1, col2, col3 = st.columns((2,50,2))
    image = Image.open('LOGO.png')
    st.image(image, caption='Involved Sports')

    with col2:
        st.header("SPORTS INVOLVED IN THE DATASET")
        st.info("""Basketball, Football, Tennis, and Cricket""")
        
file=st.file_uploader("Choose photo from computer, either Kane Williamson, Kobe Bryant, Maria Sharapova, or Ronaldo",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(150,150)
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
    class_names=['Kane Williamson','Kobe Bryant','Maria Sharapova', 'Ronaldo']
    string="Sport Celebrity : "+class_names[np.argmax(prediction)]
    st.success(string)
    
st.header("Conclusion")
st.info("""In conclusion, this activity served as an opportunity to apply the accumulated knowledge from the Final Period. The primary objectives were to train and save a model, as well as to 
        deploy the deep learning model in the cloud. By successfully completing these tasks, the intended learning outcomes were achieved. Participants were able to demonstrate their proficiency in 
        training and saving models, as well as their ability to deploy deep learning models in a cloud environment. 
        This activity effectively reinforced the essential concepts and skills related to model training, saving, and deployment, 
        contributing to the overall understanding of deep learning processes.""")
