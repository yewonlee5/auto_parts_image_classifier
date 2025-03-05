import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

class_names = {0: 'AIR COMPRESSOR',
  1: 'ALTERNATOR',
  2: 'BATTERY',
  3: 'BRAKE CALIPER',
  4: 'BRAKE PAD',
  5: 'BRAKE ROTOR',
  6: 'CAMSHAFT',
  7: 'CARBERATOR',
  8: 'COIL SPRING',
  9: 'CRANKSHAFT',
  10: 'CYLINDER HEAD',
  11: 'DISTRIBUTOR',
  12: 'ENGINE BLOCK',
  13: 'FUEL INJECTOR',
  14: 'FUSE BOX',
  15: 'GAS CAP',
  16: 'HEADLIGHTS',
  17: 'IDLER ARM',
  18: 'IGNITION COIL',
  19: 'LEAF SPRING',
  20: 'LOWER CONTROL ARM',
  21: 'MUFFLER',
  22: 'OIL FILTER',
  23: 'OIL PAN',
  24: 'OVERFLOW TANK',
  25: 'OXYGEN SENSOR',
  26: 'PISTON',
  27: 'RADIATOR',
  28: 'RADIATOR FAN',
  29: 'RADIATOR HOSE',
  30: 'RIM',
  31: 'SPARK PLUG',
  32: 'STARTER',
  33: 'TAILLIGHTS',
  34: 'THERMOSTAT',
  35: 'TORQUE CONVERTER',
  36: 'TRANSMISSION',
  37: 'VACUUM BRAKE BOOSTER',
  38: 'VALVE LIFTER',
  39: 'WATER PUMP'}

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="compressed_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Assuming your model expects 224x224 input
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image (if needed)

    # Ensure the image is in FLOAT32 type (important for TFLite)
    image = image.astype(np.float32)
    return image


# Define a function to classify the image
def classify_image(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the class with the highest probability
    class_idx = np.argmax(output_data)
    class_prob = output_data[0][class_idx] * 100  # Convert to percentage

    # Get the class name from the dictionary
    class_name = class_names.get(class_idx, "Unknown Class")

    return class_name, class_prob


# Streamlit UI
st.title("Auto Parts Image Classifier")
st.write("Transfer Learning using MobileNetV2 for image classification with 40 Classes.")

# Layout with two columns
col1, col2 = st.columns([1, 2])

# Content for col1
with col1:
    # Show class names in the description
    class_name_string = ', '.join(list(class_names.values()))
    st.write("### Available Classes:")
    st.write(f"These are the 40 classes your image could belong to:\n")
    st.text(class_name_string)

    st.write("Upload an image and the model will classify it.")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])


# Content for the right column (col2)
with col2:
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Use new parameter
        st.write("")  # Add some space

        # Run classification
        class_name, class_prob = classify_image(image)

        # Display result
        st.write(f"Predicted class: {class_name} with {class_prob:.2f}% probability")

