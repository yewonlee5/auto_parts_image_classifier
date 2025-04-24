import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from bedrock_utils import get_bedrock_explanation


# Class Name Dictionary
class_names = {0: 'AIR COMPRESSOR', 1: 'ALTERNATOR', 2: 'BATTERY', 3: 'BRAKE CALIPER',
               4: 'BRAKE PAD', 5: 'BRAKE ROTOR', 6: 'CAMSHAFT', 7: 'CARBERATOR', 8: 'COIL SPRING',
               9: 'CRANKSHAFT', 10: 'CYLINDER HEAD', 11: 'DISTRIBUTOR', 12: 'ENGINE BLOCK', 13: 'FUEL INJECTOR',
               14: 'FUSE BOX', 15: 'GAS CAP', 16: 'HEADLIGHTS', 17: 'IDLER ARM', 18: 'IGNITION COIL',
               19: 'LEAF SPRING', 20: 'LOWER CONTROL ARM', 21: 'MUFFLER', 22: 'OIL FILTER', 23: 'OIL PAN',
               24: 'OVERFLOW TANK', 25: 'OXYGEN SENSOR', 26: 'PISTON', 27: 'RADIATOR', 28: 'RADIATOR FAN',
               29: 'RADIATOR HOSE', 30: 'RIM', 31: 'SPARK PLUG', 32: 'STARTER', 33: 'TAILLIGHTS', 34: 'THERMOSTAT',
               35: 'TORQUE CONVERTER', 36: 'TRANSMISSION', 37: 'VACUUM BRAKE BOOSTER', 38: 'VALVE LIFTER',
               39: 'WATER PUMP'}

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="models/compressed_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image
    image = image.astype(np.float32)
    return image


# Function to classify the image
def classify_image(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the class with the highest probability
    class_idx = np.argmax(output_data)
    class_prob = output_data[0][class_idx] * 100  # Convert to percentage
    class_name = class_names.get(class_idx, "Unknown Class")
    return class_name, class_prob, output_data[0]


# Streamlit UI
im = Image.open('assets/car_icon.png')
st.set_page_config(layout="wide", page_title="Auto Parts Image Classifier", page_icon=im)

# Sidebar
with st.sidebar:
    st.write("""
        ## About the Model
        Finding the right car part can feel like a challenge, especially if you're not an expert! 😅 
        When an auto part breaks or falls apart, it’s tough to know its name. Even using image search can be tricky, as many auto parts have similar shapes and features! 😵 
        That’s where this app comes in! 🎉 Simply upload or take a picture of the part, and we'll tell you what it is! 📸🔍 
        """)
    # Add an expander for available classes in the sidebar
    with st.expander("40 Available Classes", expanded=False):
        class_name_string = '\n'.join(list(class_names.values()))
        st.text(class_name_string)

    st.write("""
        You can upload an image or use the camera to take a picture for classification.
    """)

    if st.button("Show EDA & Model Performance"):
        st.session_state.show_eda = True

        st.session_state["uploader_key"] += 1
        st.rerun()

    st.write("""
        ### Contact:
        """)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Yewon Lee**")
    with col2:
        st.markdown("""
            <a href="https://github.com/yewonlee5/auto_parts_image_classifier" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub" style="width: 30px; height: 30px;"/>
            </a>
            &nbsp;
            <a href="https://www.linkedin.com/in/yewonlee5" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" alt="LinkedIn" style="width: 30px; height: 30px;"/>
            </a>
            &nbsp;
            <a href="mailto:ylee52@g.ucla.edu" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/mail.png" alt="Email" style="width: 30px; height: 30px;"/>
            </a>
        """, unsafe_allow_html=True)

# Main page layout
st.title("Auto Parts Image Classifier")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1
# Image Upload Input
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key=st.session_state["uploader_key"])

# Show the spinner
if uploaded_image is not None:
    with st.spinner('Processing Image...'):
        if uploaded_image:
            image = Image.open(uploaded_image)

        # Run the model classification
        class_name, class_prob, output_data = classify_image(image)

        # Create a bar chart for predicted classes
        output_data = pd.DataFrame(output_data, columns=["Probability"])
        output_data["Class"] = list(class_names.values())
        output_data = output_data.sort_values(by="Probability", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            # Resize image for better display
            st.image(image.resize((224, 224)), caption="Uploaded Image", width=350)
        with col2:
            # Display the top 5 predictions
            plt.figure(figsize=(8, 6))
            plt.bar(output_data['Class'].head(5), output_data['Probability'].head(5), color='skyblue')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Top 5 Predictions')

            # Rotate X-axis labels
            plt.xticks(rotation=90)  # Rotate labels by 45 degrees

            st.pyplot(plt)  # Display the chart in Streamlit

            # explanation from bedrock
            explanation = get_bedrock_explanation(class_name)
            st.markdown(f"""
            <div style="border: 1px solid #DDD; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
            🧠 <b>Image Classifier Result:</b><br>
            {explanation}
            <br><sub><i>🔍 Answer generated by Amazon Bedrock</i></sub>
            </div>
            """, unsafe_allow_html=True)

        # Hide EDA & Performance after prediction
        st.session_state.show_eda = False  # Hide EDA by default after prediction

# Default view: EDA & Performance
if 'show_eda' not in st.session_state:
    st.session_state.show_eda = True

if st.session_state.show_eda:
    st.header("🚗 EDA and model performance 🛠️")
    st.write("""
    This auto parts image classifier is built using **Transfer Learning with MobileNetV2**, capable of classifying images into 40 distinct auto parts classes. 
    The model was trained on a dataset consisting of 6,917 training images, 200 validation images, and 200 test images, with a balanced distribution across all 40 classes. 
    *(Data Source: [kaggle](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes))*
    """)
    st.write("""
    The PCA analysis below highlights the top 30 features of the training data images, showing that no specific feature stands out and that common shapes (such as circles, pipes, etc.) are shared as common features. 
    This characteristic of the dataset makes it more challenging to accurately classify auto part images in real world.
    """)

    st.image("assets/1_PCA.png", caption="Principal Component Analysis")

    st.write("""
    The model demonstrates strong performance with a **93.5%** test accuracy. Additionally, you can examine the confusion matrix to gain insights into how it handles various parts.
    """)

    st.image("assets/2_confusion_matrix.png", caption="Confusion Matrix")
