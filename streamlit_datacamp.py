import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.utils import img_to_array
from keras.models import load_model

st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.image("https://www.efrei.fr/wp-content/uploads/2022/01/LOGO_EFREI-PRINT_EFREI-WEB.png")
st.sidebar.title("M1 Bioinformatics")
st.sidebar.title("Data Camp project")
st.sidebar.markdown("Groupe 4")
st.sidebar.markdown("Hajar El fakharany")
st.sidebar.markdown("Samantha Mario joy")
st.sidebar.markdown("Jean-Dylan Thomas")

def predict(testing_image):
    
    model = load_model('chest_xray.h5')
    
    image = Image.open(testing_image).convert('RGB')
    image = image.resize((224,224))
    image = img_to_array(image)
    image = image.reshape(1,224,224,3)

    result = model.predict(image)
    result = np.argmax(result, axis=-1)

    if result == 0:
        return "Patient is Normal."
    elif result == 1:
        return "Patient has Viral Pneumonia."
    else:
        return "Patient is COVID Positive."

def main():
    st.title('Covid-Pneumonia Detection')
    st.subheader('This project will predict whether a person is suffering from Covid or Viral Pneumonia using Radiograph images.')

    image = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

    if image is not None :

        #to view uploaded image
        st.image(Image.open(image))

        # Prediction
        if st.button('Result', help='Prediction'):
            st.success(predict(image))

if __name__=='__main__':
    main()
