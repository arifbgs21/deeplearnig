# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('flower_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization
    return img_array

# Streamlit app
def main():
    st.title("Flower Classification with Streamlit")

    # File uploader for image
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Gambar telah di upload.", use_column_width=True)

        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(uploaded_file)

        # Make predictions using the loaded model
        predictions = model.predict(preprocessed_image)

        # Display the prediction results
        st.subheader("Prediction Results:")
        class_labels = {0: "Lilly", 1: "Lotus", 2: "Orchid", 3: "Sunflower", 4: "Tulip"}  # Update with your class labels
        predicted_class = np.argmax(predictions)
        st.write(f"Hasil Prediksi: {class_labels[predicted_class]}")

# Run the Streamlit app
if __name__ == "__main__":
    main()