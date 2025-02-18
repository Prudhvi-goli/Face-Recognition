### **🧑‍🤖 Face Recognition System**  

This project is a **Face Recognition System** that uses **OpenCV** and **TensorFlow** to capture, train, and recognize human faces in real-time. It detects faces from a live webcam feed and identifies individuals based on a trained model.

---

## 📌 **Features**
- **Face Detection**: Detects faces using Haar Cascade classifiers.  
- **Image Capture**: Collects and stores face images for training.  
- **Model Training**: Uses TensorFlow to train a face recognition model.  
- **Real-Time Recognition**: Identifies faces live through webcam input.  

---

## 🛠️ **Requirements**
Ensure you have the following installed:

- Python 3.x  
- OpenCV (`cv2`)  
- TensorFlow  
- Keras  

To install the required packages, run:

```bash
pip install opencv-python tensorflow keras numpy
```

---

## 📂 **Project Structure**
```
.
├── dataset/                # Collected face images
│     ├── person1/
│     ├── person2/
├── keras_model.h5          # Trained face recognition model
├── haarcascade_frontalface_default.xml  # Pre-trained face detection model
├── capture_faces.py        # Collect face images
└── recognize_faces.py      # Perform real-time face recognition
```

---

## ▶️ **How to Run the Project**

### **Step 1: Collect Face Images**
Run the script to capture images for training:

```bash
python capture_faces.py
```

1. Enter the person’s name (creates a folder).  
2. Capture 300 images (or specified amount).  

---

### **Step 2: Train the Model**
Modify and use TensorFlow to train your face recognition model with the images in the `dataset/` directory.

---

## 📊 **Customize the System**
1. Add new faces by running `capture_faces.py`.  
2. Improve accuracy by increasing the dataset size.  
3. Extend recognition by updating `keras_model.h5`.  

---

## 📌 **Troubleshooting**
- Ensure the webcam is properly connected.  
- Confirm required libraries are installed.  
- Verify `keras_model.h5` exists and is correctly trained.  
