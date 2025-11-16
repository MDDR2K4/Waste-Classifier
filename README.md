# End-to-End Waste Classifier

This project is an end-to-end deep learning application that classifies images of waste into multiple categories. It includes a trained TensorFlow/Keras model and a web application built with Flask that allows users to upload an image and receive a prediction.

![Waste Classification App Screenshot]([TODO:_INSERT_A_SCREENSHOT_OF_YOUR_WEB_APP_HERE])

---

### Table of Contents
* [Project Overview](#project-overview)
* [Problem Statement](#problem-statement)
* [Dataset](#dataset)
* [Technology Stack](#technology-stack)
* [Project Workflow](#project-workflow)
* [Setup and Installation](#setup-and-installation)
* [How to Run](#how-to-run)
* [Model Performance and Results](#model-performance-and-results)
* [Future Improvements](#future-improvements)
* [Author](#author)
* [License](#license)

---

### Project Overview

This repository contains the code and resources for building a complete waste classification system. The core of the project is a Convolutional Neural Network (CNN) built using transfer learning with MobileNetV2. The trained model is served via a Flask API, providing a simple and intuitive web interface for real-time predictions.

---

### Problem Statement

The goal is to automatically classify waste into one of the following predefined categories, helping to streamline recycling and waste management processes:
- Battery
- Biological
- Cardboard
- Clothes
- Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash

---

### Dataset

The model was trained on the **Garbage Classification V2** dataset.
- **Source:** [Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- **Description:** The dataset contains thousands of images organized into folders, with each folder representing a distinct waste category.

---

### Technology Stack

- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow, Keras
- **Data Handling:** NumPy, Pillow
- **Development Environment:** Jupyter Notebook, VS Code
- **Deployment (planned):** Docker, Cloud Platform (e.g., Heroku, AWS)

---

### Project Workflow

The project follows a standard machine learning project lifecycle:

1.  **Data Collection & Preprocessing:** Images are loaded from the dataset directory, resized, and batched using TensorFlow's `image_dataset_from_directory`.
2.  **Model Building & Training:** A pre-trained MobileNetV2 model is used as a base. The top layers are replaced with a new classifier, which is then trained on our specific dataset.
3.  **Model Evaluation:** The model's performance is evaluated on a validation set to check for accuracy and prevent overfitting.
4.  **Model Saving:** The trained model is saved to an `.h5` file.
5.  **Flask API Development:** A Flask server is built to load the saved model and expose a prediction endpoint.
6.  **Frontend Development:** A simple HTML/CSS interface is created to allow users to upload images and view the prediction.

---

### Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MDDR2K4/Week-1
    cd Week-1
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

### How to Run

There are two main parts to this project: training the model and running the web application.

**1. Training the Model**
- The model training and experimentation code is located in the `notebooks/` directory.
- To run the notebook, start the Jupyter server from your activated virtual environment:
  ```bash
  jupyter notebook
  ```
- Open `waste_classification.ipynb` and run the cells to train the model. The final trained model will be saved in the `models/` directory.

**2. Running the Web Application**
- The Flask application is the `app/app.py` script.
- To start the server, run the following command from the root directory of the project:
  ```bash
  python app/app.py
  ```
- Open your web browser and navigate to: `http://127.0.0.1:5000`

---

### Model Performance and Results

The model was trained for **10 epochs** and achieved the following performance on the validation set:

-   **Final Validation Accuracy:91.52** 
-   **Final Validation Loss:0.2628** 

**Training History:**

Below is the plot showing the model's accuracy and loss over the training epochs.

[images/Screenshot 2025-11-16 202654.png]

![Training History]([PATH_TO_YOUR_PLOT_IMAGE])

---
### Output (Screenshots)


**1. Main Application Page**
![App Home Page]([PATH_TO_YOUR_HOME_PAGE_SCREENSHOT])

**2. Prediction Result Page**
![App Prediction Page]([PATH_TO_YOUR_PREDICTION_RESULT_SCREENSHOT])

### Future Improvements

- **Fine-Tuning:** Unfreeze more layers of the base MobileNetV2 model and fine-tune with a lower learning rate to potentially increase accuracy.
- **Experiment with other models:** Try other pre-trained architectures like ResNet50 or VGG16.
- **Deployment:** Containerize the application using Docker and deploy it to a cloud service like Heroku or AWS for public access.
- **Improve UI:** Enhance the web interface with better styling and user feedback.

---

### Author

- **M Deeraj D Rao**
- **GitHub:`https://github.com/MDDR2K4`** 
- **LinkedIn:`https://www.linkedin.com/in/mdeerajdrao`** 


### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
