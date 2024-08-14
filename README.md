
# Face Mask Detection Model

## Overview
This project is a deep learning-based face mask detection system, designed to predict whether a person in an image is wearing a face mask or not. The system is built using TensorFlow and Keras, leveraging a Convolutional Neural Network (CNN) model.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used for this project consists of images with two classes: 
- **With Mask**
- **Without Mask**

Ensure you have a properly labeled dataset before training the model. You can use publicly available datasets or create your own by collecting images.

## Model Architecture
The model is a Convolutional Neural Network (CNN) designed to classify images into two categories. The architecture includes the following layers:
- Convolutional layers for feature extraction
- MaxPooling layers to reduce dimensionality
- Fully connected layers for classification

The model is trained using the `Adam` optimizer and `categorical_crossentropy` loss function.

## Installation
To run this project, you need to have Python 3.x installed along with the following dependencies:
```bash
pip install tensorflow keras numpy matplotlib
```

## Usage
1. **Training the Model**: 
    - Place your dataset in the appropriate directory.
    - Modify the paths in the notebook to point to your dataset.
    - Run the notebook cells to train the model.

2. **Making Predictions**:
    - Load the trained model using `load_model('face_mask.h5')`.
    - Use the provided functions to preprocess images and make predictions.

Example:
```python
model = load_model('face_mask.h5')
predict_and_display('path_to_image.jpg')
```

## Results
The trained model achieves high accuracy in distinguishing between images with and without face masks. Below is an example of the model's prediction:

![Example Prediction](example_prediction.png)

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any changes or improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
