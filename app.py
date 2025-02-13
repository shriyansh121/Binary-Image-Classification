import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

model = load_model('model3.h5')

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_image(image):
    image = image.resize((224,224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0) 
    image = image / 255.
    result = model.predict(image)
    #predicted_class = np.argmax(result, axis=1)  
    return result

def main():
    st.title("Image Classification App for Room and Street Images")
    st.write("Upload an image and the model will classify it")

    image_file = st.file_uploader("Upload an image: ", type = ['jpg', 'jpeg', 'png'])
    if image_file is not None:
        image = load_image(image_file)
        st.image(image, caption = 'Uploaded Image', use_container_width=True)
        if st.button('Predict'):
            prediction = predict_image(image)
            if prediction<0.5:
                st.write("Image is of house")
            else:
                st.write("Image is of Street")
            st.write(prediction)
        else:
            st.write("Click the predict button to classify the image.")


if __name__ == '__main__':
    main()

