import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained weather classification model
def load_model():
    try:
        st.write("Predicting...")
        model = tf.keras.models.load_model('WeatherClassification.h5')
        st.success("Predicted Successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(image):
    size = (150, 150)
    resized_image = image.resize(size)  # Resize the image to match the model input size
    normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values
    preprocessed_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    return preprocessed_image

# UI setup
st.markdown("""
    <h1 style='text-align: center; color: blue;'>Weather Image Classification</h1>
    <div style="text-align: center;">Group 12 - Vinoya, Michael Jr. A. | Villanueva, Marc Joseph</div>
    <h3 style="text-align: center; color: blue; ">(Cloudy | Rain | Shine | Sunrise)</h3>
    """, unsafe_allow_html=True)

# Upload image
uploaded_image = st.file_uploader(
    label="Choose file:",
    type=["jpg", "png", "jpeg"]
)

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the model
    model = load_model()

    if model is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)

        # Define weather categories
        weather_conditions = ['Cloudy', 'Rainy', 'Shine', 'Sunrise']

        # Compute the probabilities
        probabilities = prediction[0] * 100

        # Determine the predicted weather condition
        predicted_condition = weather_conditions[np.argmax(prediction)]

        # Display the probabilities for each condition
        st.write("Prediction Probabilities:")
        for condition, probability in zip(weather_conditions, probabilities):
            st.write(f"{condition}: {probability:.2f}%")

        # Display the predicted condition
        st.write("Predicted Weather Condition:", predicted_condition)
    else:
        st.write("Failed to load the model.")
else:
    st.write("Please upload an image to get a prediction.")
