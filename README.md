# 🖼️ Image Classification using CNN with Image Processing Techniques

This project demonstrates how core image processing libraries and deep learning models can be combined to solve a real-world image classification task. Using tools like **OpenCV**, **Pillow (PIL)**, and **NumPy**, this project applies a preprocessing pipeline to satellite/aerial-style images, which are then classified using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.

The model is trained to classify images into five land-use categories:
- 🌲 Forest  
- 🏭 Industrial  
- 🌾 Permanent Crop  
- 🏘️ Residential  
- 🌊 River  

---

## 📂 Project Structure
```
image-classification/
│
├── data/                          # Image dataset organized by class
│   ├── Forest/
│   ├── Industrial/
│   ├── PermanentCrop/
│   ├── Residential/
│   └── River/
│
├── cnn_image_classification.ipynb # Main Jupyter notebook with preprocessing & CNN
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitkeep                       # Maintains directory in Git
```

---

## 🧰 Libraries & Tools Used

### 🖼️ Image Processing
- `OpenCV (cv2)` – Reading and resizing images
- `Pillow (PIL)` – Image format handling
- `NumPy` – Pixel scaling and array manipulation

### 🧠 Deep Learning
- `TensorFlow / Keras` – CNN model building and training
- `Matplotlib` – Visualization of training history
- `Scikit-learn` – Confusion matrix and classification report

---

## 🖼️ Image Preprocessing Pipeline

Each image undergoes the following steps before training:

- Loaded using `cv2.imread()` and converted to RGB
- Resized to a fixed shape (150x150 pixels)
- Normalized to the range [0, 1] using NumPy
- Labels are one-hot encoded using `to_categorical`

These steps ensure consistent input format for the CNN and help improve training stability.

---

## 🧠 CNN Model Architecture

The network includes:

- **Conv2D** + **ReLU** activation
- **MaxPooling2D** for spatial downsampling
- **Dropout** for regularization
- **Flatten** and **Dense** layers
- **Softmax** output layer for multi-class classification

> **Loss:** Categorical Crossentropy  
> **Optimizer:** Adam  
> **Metric:** Accuracy

---

## 📦 Dataset
The dataset used in this project was sourced from the EuroSAT dataset (https://paperswithcode.com/dataset/eurosat), which contains Sentinel-2 satellite imagery covering 10 land use and land cover classes. For this project, a subset of five classes was selected: Forest, Industrial, PermanentCrop, Residential, and River. The dataset is publicly available and widely used for benchmarking image classification tasks.

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/santosh519/image-classification.git
   cd image-classification

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Launch the notebook**
   ```bash
   jupyter notebook cnn_image_classification.ipynb

The data/ folder already contains the necessary training images, organized by class.

---

## 👤 Author & Contact
Santosh Adhikari

Email: santadh2015@gmail.com

GitHub: @santosh519

Thank you for reviewing! Feedback and contributions are welcome.
