import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("image_classify.h5")


class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


st.title("Image Classification Web App")
st.write("Upload an image to classify it")


uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)


    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100


    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")