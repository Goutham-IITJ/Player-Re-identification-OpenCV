# ⚽ Player Re-Identification in Sports Footage

This project performs **player detection and re-identification** in sports videos. It detects players using a fine-tuned YOLOv8 model (`best.pt`) and maintains consistent IDs across frames using **feature-based Re-ID tracking** powered by **Norfair** and a **ResNet50** feature extractor.

---

## 📁 Project Structure

```
Player-Re-identification-OpenCV/
├── best.pt                     # YOLOv8 model checkpoint (provided)
├── 15sec_input_720p.mp4        # Input sports video
├── outputs/                    # Folder for output videos
├── main.py                     # Main pipeline script
├── yolo_detector.py            # YOLOv8 detection module
├── feature_extractor.py        # Re-ID feature extractor
├── reid_tracker.py             # Tracker using cosine similarity
├── helpers.py                  # Utility functions (e.g., crop + draw)
├── player_reid_env/            # Python virtual environment
└── README.md                   # This file
```

---

## 🚀 Features

- 🔍 **YOLOv8** for accurate player detection
- 🎯 **Re-identification (Re-ID)** using ResNet-50 feature extraction
- 🔁 **ID consistency** maintained even when players disappear and reappear
- 🟢 Clean tracking visualization with bounding boxes and IDs
- 🧠 Fully modular architecture for easy extension and customization
- ⚡ Optimized for real-time processing

---

## 🧩 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Goutham-IITJ/Player-Re-identification-OpenCV.git
cd Player-Re-identification-OpenCV
```

### 2. Create Virtual Environment

```bash
python -m venv player_reid_env

# On Windows
player_reid_env\Scripts\activate

# On macOS/Linux
source player_reid_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install ultralytics norfair opencv-python timm torch torchvision
```

### 4. Run the Project

```bash
python main.py
```

---

## 📋 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- Norfair
- timm
- torchvision

---

## 🎥 Usage

1. Place your input video in the project directory
2. Update the video path in `main.py` if needed
3. Run the script to process the video
4. Check the `outputs/` folder for the processed video with player tracking

---

## 🔧 Configuration

You can modify the following parameters in `main.py`:

- Detection confidence threshold
- Re-ID similarity threshold
- Video input/output paths
- Tracking parameters

---

## 📊 How It Works

1. **Detection**: YOLOv8 detects players in each frame
2. **Feature Extraction**: ResNet50 extracts appearance features from detected players
3. **Re-Identification**: Cosine similarity matching maintains consistent IDs
4. **Tracking**: Norfair handles track management and interpolation
5. **Visualization**: Results are drawn on frames and saved as output video

---

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

## 📝 License

This project is open source. Please check the repository for license details.

---

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for object detection
- Norfair for tracking framework
- ResNet50 for feature extraction
- OpenCV for video processing
