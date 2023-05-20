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
st.image(image, caption='Sports Celebrity Classifier - James Brian Antonio')

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
st.info("""In conclusion, I have successfully applied all the learnings from the Final Period in this activity. 
The main objectives were to train and save a model, as well as to deploy the deep learning model in the cloud. By accomplishing these tasks, I have demonstrated my 
ability to fulfill the intended learning outcomes.

Throughout the activity, I learned how to train a model and save its weights or the entire model itself. Additionally, I gained hands-on experience in deploying a deep 
learning model in a cloud environment. This involved utilizing tools like GitHub and Streamlit to make the model accessible and interactive.

By completing this activity, I have solidified my understanding of the processes involved in training, saving, and deploying deep learning models. These skills will be 
valuable in future projects and contribute to my overall proficiency in deep learning.""")

st.info("""Basketball, Football, Tennis, and Cricket""")
