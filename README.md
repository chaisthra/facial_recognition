# Smart Face Recognition System 🎯

A modern face recognition system built with Python, OpenCV, and Streamlit featuring real-time detection, face registration, and attendance tracking.

## ✨ Features

- **Real-time Face Recognition**
  - Live camera feed recognition
  - Multiple face detection
  - Confidence threshold adjustment
  - Attendance tracking

- **Face Registration**
  - Camera capture
  - Image upload
  - Face preview
  - Easy deletion

- **Modern UI**
  - Cyberpunk theme
  - Responsive design
  - Real-time feedback
  - Grid view of registered faces

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-recognition-system.git
   cd face-recognition-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure
```
face-recognition-system/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── README.md          # Documentation
├── known_faces/       # Stored face images
└── attendance/        # Attendance records
```

## 🔧 Usage

1. **Add New Face**
   - Enter person's name
   - Either upload photo or capture via camera
   - Verify face detection
   - Save face

2. **Recognition**
   - Start camera
   - Adjust sensitivity if needed
   - View real-time detections
   - Check attendance records

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request
