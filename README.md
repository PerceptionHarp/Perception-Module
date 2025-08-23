# Gender Classification
This repository contains the Gender Classification System developed for a Humanoid Assistance Robot.
The system leverages a Convolutional Neural Network (CNN) to classify gender in real-time from a live video feed captured by the RealSense camera. By integrating this deep learning model, the robot is able to recognize gender with high accuracy, allowing it to deliver personalized assistance and enhance the overall human-robot interaction experience.

## Features 🚀

* Real-time gender classification using a trained deep learning model.

* Optimized with TensorFlow Lite / YOLOv8 (depending on your final setup).

* Integrated with OpenCV for video streaming and frame processing.

* Can be extended for age detection, emotion recognition, and action recognition for more intelligent interaction.

## Tech Stack 🛠️

* Language: Python 3.x

* Frameworks & Libraries:

* Ultralytics YOLOv8 / TensorFlow Lite

* OpenCV

* NumPy

## Dataset

https://drive.google.com/drive/folders/1LGRt5SGPTJkDEqPhMIYzH_9o7aJq62US


## 🖥️ How to Use the `save_live_gender_vlc.py` Script

Follow these steps to integrate everything and run the real-time gender classification system.

## 0) Requirements

* Python 3.8+
* Webcam
* (Optional) VLC player to preview saved video
* (Optional) CUDA-enabled GPU for faster training/inference

---

## 1) Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

*(Optional) create a virtual environment*

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

## 2) Install Dependencies

```bash
pip install ultralytics opencv-python
```

> If you run into display issues on servers without GUI, add:
> `pip install opencv-python-headless`

---

## 3) (Optional) Train Your Own Model

**Prepare dataset (two classes: `male`, `female`):**

```
dataset/
 ├─ train/
 │   ├─ male/
 │   └─ female/
 └─ val/
     ├─ male/
     └─ female/
```

**Train with YOLOv8 classification (CNN-based):**

```bash
yolo classify train model=yolov8n-cls.pt data=dataset epochs=50 imgsz=224
```

**Your best weights will be saved to:**

```
runs/classify/train/weights/best.pt
```

---

## 4) Configure the Script

Open `save_live_gender_vlc.py` and set these values:

```python
MODEL_PATH  = "runs/classify/train/weights/best.pt"  # your trained model
OUTPUT_PATH = "gender_output.mp4"                    # saved video path
CODEC       = "mp4v"                                 # use "mp4v" for .mp4, "XVID" for .avi
```

> If your webcam isn’t at index `0`, change:

```python
cap = cv2.VideoCapture(0)  # try 1, 2, etc., if 0 fails
```

---

## 5) Run Live Gender Classification

```bash
python save_live_gender_vlc.py
```

**Controls**

* A window appears: “Live Gender (press q to quit)”
* Face boxes + predicted **gender** and **confidence** are drawn
* **FPS** is shown (smoothed)
* Press **`q`** to stop and save the video

---

## 6) Check the Saved Output

Default output file:

```
gender_output.mp4
```

Play it in VLC:

```bash
vlc gender_output.mp4
```

> On Windows, if `vlc` isn’t on PATH, open VLC and drag-drop the file.

---

## 7) Project Structure (Recommended)

```
<repo-root>/
├─ save_live_gender_vlc.py
├─ README.md
├─ dataset/                 # only if you train
│  ├─ train/
│  │  ├─ male/
│  │  └─ female/
│  └─ val/
│     ├─ male/
│     └─ female/
└─ runs/                    # created by YOLO after training
   └─ classify/
      └─ train/
         └─ weights/
            └─ best.pt
```





## Results

https://youtu.be/_wrjzWkiH2A

## Applications in Humanoid Robot 🤖

* Receptionist Robots – greeting and addressing people naturally.

* Assistance Robots – tailoring services depending on classification.

* Elderly Care Robots – enhancing trust and communication.

## Acknowledgments 🙌

* Ultralytics YOLOv8

* TensorFlow Lite

* Open-source AI community
