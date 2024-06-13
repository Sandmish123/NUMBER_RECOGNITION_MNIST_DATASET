# import streamlit as st
# import pickle
# import tensorflow as tf
# import numpy as np

# # Function to load the model from the pickle file
# @st.cache(allow_output_mutation=True)  # Cache the model to avoid loading it multiple times
# def load_model():
#     with open('mnist_cnn_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model

# # Function to preprocess image
# def preprocess_image(image):
#     # Preprocess as needed (resize, normalize, etc.)
#     return image

# # Main function to run the Streamlit app
# def main():
#     st.title('MNIST Digit Recognition')
#     st.write('Upload a handwritten digit image for prediction')
    
#     # File uploader to upload an image
#     uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    
#     # Check if an image has been uploaded
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = tf.image.decode_image(uploaded_file.read(), channels=1)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
        
#         # Make predictions
#         model = load_model()
#         processed_image = preprocess_image(image)
#         prediction = model.predict(np.expand_dims(processed_image, axis=0))
#         predicted_class = np.argmax(prediction)



# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io

# # Function to load the model from the pickle file
# @st.cache(allow_output_mutation=True)  # Cache the model to avoid loading it multiple times
# def load_model():
#     # Assuming 'mnist_cnn_model.pkl' is a serialized TensorFlow model
#     model = tf.keras.models.load_model('mnist_cnn_model.h5')
#     return model

# # Function to preprocess image
# def preprocess_image(image):
#     # Resize image to 28x28 (assuming your model expects this size)
#     image = image.resize((28, 28))
#     # Convert to grayscale (if necessary)
#     image = image.convert('L')
#     # Convert image to numpy array
#     image = np.array(image)
#     # Normalize pixel values to 0-1
#     image = image / 255.0
#     # Reshape image for model input (add batch dimension)
#     image = np.expand_dims(image, axis=0)
#     # Add channel dimension if image shape is (28, 28) instead of (28, 28, 1)
#     if image.shape[-1] == 2:
#         image = np.expand_dims(image, axis=-1)
#     return image

# # Main function to run the Streamlit app
# def main():
#     st.title('MNIST Digit Recognition')
#     st.write('Upload a handwritten digit image for prediction')
    
#     # File uploader to upload an image
#     uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    
#     # Check if an image has been uploaded
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
        
#         # Make predictions
#         model = load_model()
#         processed_image = preprocess_image(image)
#         prediction = model.predict(processed_image)
#         predicted_class = np.argmax(prediction)
        
#         st.write(f'Prediction: {predicted_class}')

# if __name__ == '__main__':
#     main()

# #         st.write(f'Prediction: {predicted_class}')







import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the model from the HDF5 file
@st.cache_data(allow_output_mutation=True)  # Updated caching command
def load_model():
    # Load the model in HDF5 format
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

# Function to preprocess image
def preprocess_image(image):
    # Resize image to 28x28 (assuming your model expects this size)
    image = image.resize((28, 28))
    # Convert to grayscale (if necessary)
    image = image.convert('L')
    # Convert image to numpy array
    image = np.array(image)
    # Normalize pixel values to 0-1
    image = image / 255.0
    # Reshape image for model input (add batch dimension)
    image = np.expand_dims(image, axis=0)
    # Add channel dimension if image shape is (28, 28) instead of (28, 28, 1)
    if image.shape[-1] == 2:
        image = np.expand_dims(image, axis=-1)
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
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Make predictions
        model = load_model()
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        
        st.write(f'Prediction: {predicted_class}')

if __name__ == '__main__':
    main()


# if __name__ == '__main__':
#     main()
