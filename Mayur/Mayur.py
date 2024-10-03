import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image

class_names = ["apple apple scab", "apple black rot", "apple cedar apple rust", "apple healthy", "blueberry healthy", 
               "cherry including sour powdery mildew", "cherry including sour healthy", "corn maize cercospora leaf spot gray leaf spot", 
               "corn maize common rust", "corn maize northern leaf blight", "corn maize healthy", "grape black rot", 
               "grape esca black measles", "grape leaf blight isariopsis leaf spot", "grape healthy", 
               "orange haunglongbing citrus greening", "peach bacterial spot", "peach healthy", 
               "pepper bell bacterial spot", "pepper bell healthy", "potato early blight", "potato late blight", 
               "potato healthy", "raspberry healthy", "soybean healthy", "squash powdery mildew", 
               "strawberry leaf scorch", "strawberry healthy", "tomato bacterial spot", "tomato early blight", 
               "tomato late blight", "tomato leaf mold", "tomato septoria leaf spot", "tomato spider mites two spotted spider mite", 
               "tomato target spot", "tomato tomato yellow leaf curl virus", "tomato tomato mosaic virus", "tomato healthy"]

# Page Title
st.set_page_config(page_title="Plant Disease Detection")
st.title("Plant Disease Detection")
st.markdown("---")

# Load the TFLite model
tflite_interpreter = tf.lite.Interpreter(model_path='Mayur\model (2).tflite')
tflite_interpreter.allocate_tensors()

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def preprocess_image(image, target_size=(200, 200)):
    """Preprocess the input image by resizing it to the target size."""
    image = Image.fromarray(image).resize(target_size)
    image = np.array(image)
    return image

def get_predictions(input_image):
    output_details = tflite_interpreter.get_output_details()
    set_input_tensor(tflite_interpreter, input_image)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis=0)
    pred_class = class_names[tflite_model_prediction]
    return pred_class

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((224, 224)) 
    st.image(img)

  
    img_array = np.array(img)
    img_array = preprocess_image(img_array)  
    img_array = tf.expand_dims(img_array, 0)  

if st.button("Get Predictions"):
    if uploaded_file is not None:
        suggestion = get_predictions(input_image=img_array)
        if suggestion == 'no disease':
            st.success(suggestion)
        else:
            st.success(f"Disease detected: {suggestion}")
    else:
        st.error("Please upload an image.")
