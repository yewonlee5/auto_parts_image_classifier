# 🚗 Auto Parts Image Classifier 🛠️

The wide variety of automotive parts and their similar features can make it challenging for non-experts to identify the right parts, especially when they have a broken or fallen-off part but aren’t sure which one it is. Many parts share common characteristics, making it difficult to distinguish between them.

To address this, I developed this auto parts image classifier using **Transfer Learning with MobileNetV2**. The model is capable of classifying images into 40 distinct auto parts classes. It was trained on a dataset consisting of 6,917 training images, 200 validation images, and 200 test images, with a balanced distribution across all 40 classes. *(Data Source: [kaggle](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes))*

The model achieves a **93.5%** test accuracy, helping non-experts quickly figure out which part they need to purchase. You can explore the model further through the link below.

[Explore the Classifier Here](https://autopartsimageclassifier.streamlit.app/)
![GIF](assets/auto_classifier_screenshot.gif)
