import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.utils import img_to_array
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.image("https://www.efrei.fr/wp-content/uploads/2022/01/LOGO_EFREI-PRINT_EFREI-WEB.png")
st.sidebar.title("M1 Bioinformatics")
st.sidebar.title("Data Camp project")
st.sidebar.markdown("Groupe 4")
st.sidebar.markdown("Hajar El fakharany")
st.sidebar.markdown("Samantha Mario joy")
st.sidebar.markdown("Jean-Dylan Thomas")


def predict(testing_image):
    
    model = load_model(r"C:\Users\saman\OneDrive\Documents\EFREI\Datacamp\chest_xray.h5")
    image = Image.open(testing_image).convert('RGB')
    image = image.resize((224,224))
    x = img_to_array(image)
    x=np.expand_dims(x, axis=0)
    image_data=preprocess_input(x)
    classes=model.predict(image_data)
    result=int(classes[0][0])



    if result==0:
            # print("Person is Affected By PNEUMONIA")
        return "Person has Pneumonia"
    else:
        return "Result is Normal"
            # print("Result is Normal")



def main():
    st.title('Pneumonia Detection')
    st.subheader('This project will predict whether the image is a Normal chest X-ray or a Pneumonia chest X-ray.')

    image = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

    if image is not None :

        #to view uploaded image
        st.image(Image.open(image))

        # Prediction
        if st.button('Result', help='Prediction'):
            st.success(predict(image))

if __name__=='__main__':
    main()
