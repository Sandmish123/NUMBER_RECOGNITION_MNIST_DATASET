import streamlit as st
import pickle
import tensorflow as tf
import numpy as np

# Function to load the model from the pickle file
@st.cache(allow_output_mutation=True)  # Cache the model to avoid loading it multiple times
def load_model():
    with open('mnist_cnn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess image
def preprocess_image(image):
    # Preprocess as needed (resize, normalize, etc.)
    return image

# Main function to run the Streamlit app
def main():
    st.title('MNIST Digit Recognition')
    st.write('Upload a handwritten digit image for prediction')
    
    # File uploader to upload an image
    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    
    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        image = tf.image.decode_image(uploaded_file.read(), channels=1)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Make predictions
        model = load_model()
        processed_image = preprocess_image(image)
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        predicted_class = np.argmax(prediction)
        
        st.write(f'Prediction: {predicted_class}')

if __name__ == '__main__':
    main()
