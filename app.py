import cv2
import streamlit as st
import face_recognition
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import csv
import time
import base64
from auth import show_auth_page
import json
import shutil

# Set page configuration first - this must be the first Streamlit command
st.set_page_config(
    page_title="Facial Recognition System",
    page_icon="👤",
    layout="wide"
)

# Add at the top of app.py after imports
MAX_CAMERA_WAIT = 30
FRAME_TIMEOUT = 5.0

# Function to get background image
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Try to load background image
try:
    bg_image = get_base64_of_image("assets/background.jpeg")
    background_style = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                         url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Center content with padding */
    .block-container {{
        max-width: 1400px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: auto !important;
    }}

    /* Adjust main content padding */
    .main .block-container {{
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }}

    /* Center header */
    .stApp header {{
        background-color: transparent !important;
    }}

    /* Center tabs */
    .stTabs {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem 0;
    }}
    </style>
    """
except Exception:
    # Fallback to gradient background
    background_style = """
    <style>
    .stApp {
        background: linear-gradient(45deg, #1a1a1a, #0a192f);
    }
    
    .block-container {
        max-width: 1000px !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
        margin: auto !important;
    }

    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
    """

# Apply styles after page config
st.markdown(background_style, unsafe_allow_html=True)

# Custom CSS with cyberpunk styling
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp > header {
        background-color: transparent !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        padding: 1rem 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: #00ff9d !important;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
    }

    /* Card styling */
    .stCard {
        background-color: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(0, 255, 157, 0.2);
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }

    /* Text colors and sizes */
    h1 {
        color: #00ff9d !important;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        font-size: 3rem !important;
        margin-bottom: 2rem !important;
    }

    h2 {
        color: #00ff9d !important;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        font-size: 2.2rem !important;
        margin: 2rem 0 !important;
    }

    h3 {
        color: #00ff9d !important;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        font-size: 1.8rem !important;
        margin-bottom: 1.5rem !important;
    }

    p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }

    /* Button styling */
    .stButton > button {
        border: 2px solid #00ff9d !important;
        color: #00ff9d !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        margin: 1rem 0 !important;
    }

    .stButton > button:hover {
        background-color: rgba(0, 255, 157, 0.2) !important;
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.5);
        transform: translateY(-2px);
    }

    /* Input fields */
    .stTextInput input, .stTextInput textarea {
        color: black !important;
        background-color: white !important;
        font-size: 1.1rem !important;
        padding: 0.5rem !important;
    }

    .stTextInput > div {
        background-color: white !important;
        border: 1px solid rgba(0, 255, 157, 0.2) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }

    /* Slider */
    .stSlider {
        opacity: 0.8;
        padding: 1rem 0 !important;
    }

    .stSlider > div {
        margin: 1rem 0 !important;
    }

    /* Image container */
    .face-image-container {
        border: 2px solid rgba(0, 255, 157, 0.3);
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
        background-color: rgba(0, 0, 0, 0.4);
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.2);
    }

    /* Column spacing */
    .row-widget.stHorizontal {
        gap: 2rem !important;
    }

    /* Camera feed container */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }

    /* Detection list */
    .element-container {
        margin: 1rem 0 !important;
    }

    /* File uploader */
    .stUploadButton {
        margin: 1rem 0 !important;
    }

    .uploadedFile {
        color: black !important;
        background-color: white !important;
    }

    /* Checkbox text */
    .stCheckbox label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Add these constants at the top of the file
CAMERA_STOPPED = False
MAX_RETRIES = 3

# Create necessary folders if they don't exist
def create_folders():
    Path("known_faces").mkdir(exist_ok=True)
    Path("attendance").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

# 1. Add caching for known faces
@st.cache_data
def load_known_faces(username):
    known_face_encodings = []
    known_face_names = []
    
    try:
        # Load user's faces from JSON
        with open('data/users.json', 'r') as f:
            data = json.load(f)
            user = next((u for u in data['users'] if u['username'] == username), None)
            
            if user and user['known_faces']:
                print(f"Found faces in JSON: {len(user['known_faces'])}")  # Debug print
                
                for face_data in user['known_faces']:
                    try:
                        face_path = Path(face_data['path'])
                        print(f"Processing face: {face_path}")  # Debug print
                        
                        if face_path.exists():
                            print(f"Face file exists: {face_path}")  # Debug print
                            face_image = face_recognition.load_image_file(str(face_path))
                            face_locations = face_recognition.face_locations(face_image, model="hog")
                            
                            if face_locations:
                                face_encoding = face_recognition.face_encodings(face_image, [face_locations[0]])[0]
                                known_face_encodings.append(face_encoding)
                                known_face_names.append(face_data['name'])
                                print(f"Successfully loaded face: {face_data['name']}")  # Debug print
                    except Exception as e:
                        print(f"Error processing face: {str(e)}")  # Debug print
                        pass
            else:
                print(f"No faces found for user: {username}")  # Debug print
    except Exception as e:
        print(f"Error loading faces: {str(e)}")  # Debug print
    
    return known_face_encodings, known_face_names

def record_attendance(username, person_name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('data/users.json', 'r+') as f:
        data = json.load(f)
        for user in data['users']:
            if user['username'] == username:
                user['attendance_records'].append({
                    'name': person_name,
                    'timestamp': timestamp
                })
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
                break

def initialize_camera():
    """Initialize camera with basic settings"""
    try:
        # Try different camera indices
        for camera_index in [0, 1]:
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                # Set camera properties
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
                camera.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Adjust brightness
                
                # Test camera and wait for it to initialize
                for _ in range(10):  # Give camera time to warm up
                    ret, frame = camera.read()
                    if ret and frame is not None and not np.all(frame == 0):
                        return camera
                    time.sleep(0.1)
                camera.release()
        
        st.error("No working camera found")
        return None
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return None

def safe_camera_release(camera):
    """Safely release camera resources"""
    try:
        if camera is not None:
            if camera.isOpened():
                camera.release()
            cv2.destroyAllWindows()
    except Exception:
        pass

# 2. Optimize face detection in real-time recognition
def process_frame(frame, scale=0.25):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find faces in smaller frame
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Scale back up face locations
    face_locations = [(int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)) 
                     for (top, right, bottom, left) in face_locations]
    
    return face_locations, face_encodings

def save_face_data(username, face_name, face_path):
    try:
        # Create user-specific directory
        user_face_dir = Path(f"known_faces/{username}")
        user_face_dir.mkdir(parents=True, exist_ok=True)
        
        # Save face image in user's directory
        new_face_path = user_face_dir / f"{face_name}.jpg"
        
        # Copy instead of move to avoid permission issues
        if Path(face_path).exists():
            shutil.copy2(face_path, new_face_path)
            Path(face_path).unlink()  # Remove original after copy
            
            # Update JSON
            with open('data/users.json', 'r+') as f:
                data = json.load(f)
                for user in data['users']:
                    if user['username'] == username:
                        user['known_faces'].append({
                            'name': face_name,
                            'path': str(new_face_path)
                        })
                        f.seek(0)
                        json.dump(data, f, indent=4)
                        f.truncate()
                        break
            
            st.success(f"Face saved successfully: {face_name}")
            return True
    except Exception as e:
        st.error(f"Error saving face: {str(e)}")
        return False

def load_user_faces(username):
    with open('data/users.json', 'r') as f:
        data = json.load(f)
        for user in data['users']:
            if user['username'] == username:
                return user['known_faces']
    return []

def display_stored_faces(username):
    st.markdown("<h2>Registered Faces</h2>", unsafe_allow_html=True)
    
    # Load user's faces from JSON
    with open('data/users.json', 'r') as f:
        data = json.load(f)
        user = next((u for u in data['users'] if u['username'] == username), None)
        if user and user['known_faces']:
            face_cols = st.columns(4)
            
            for idx, face_data in enumerate(user['known_faces']):
                face_path = Path(face_data['path'])
                if face_path.exists():
                    with face_cols[idx % 4]:
                        st.markdown("""
                        <div class="face-image-container">
                        """, unsafe_allow_html=True)
                        image = cv2.imread(str(face_path))
                        if image is not None:
                            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(rgb_image, caption=face_data['name'])
                            if st.button(f"🗑️ Remove", key=f"del_{idx}"):
                                # Remove face from JSON and file system
                                face_path.unlink()
                                user['known_faces'].pop(idx)
                                with open('data/users.json', 'w') as f:
                                    json.dump(data, f, indent=4)
                                st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No faces registered yet")

def main():
    # Check authentication
    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = None
    
    if not st.session_state.auth_status:
        show_auth_page()
        return
    
    # Add logout button in sidebar
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.auth_status = None
            st.session_state.username = None
            st.rerun()
    
    # Main title with modern styling
    st.markdown("<h1>Smart Face Recognition System</h1>", unsafe_allow_html=True)
    
    # Create tabs instead of selectbox
    tab1, tab2 = st.tabs(["📸 Add New Face", "🔍 Face Recognition"])
    
    with tab1:
        st.markdown("<h2>Register New Face</h2>", unsafe_allow_html=True)
        
        # Create two columns with modern cards
        col1, col2 = st.columns(2)
        
        # Name input first
        person_name = st.text_input("Enter person's name")
        
        with col1:
            st.markdown("""
            <div class="stCard">
                <h3>Upload Photo</h3>
                <p>Upload a clear photo of the person's face</p>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
            
            # Handle uploaded file
            if uploaded_file is not None and person_name:
                try:
                    # Read and process uploaded image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Display uploaded image
                    st.image(rgb_image, caption="Uploaded Image")
                    
                    # Detect faces in uploaded image
                    face_locations = face_recognition.face_locations(rgb_image)
                    
                    if face_locations:
                        if st.button("Save Uploaded Face"):
                            # Save the image with face cropping
                            top, right, bottom, left = face_locations[0]
                            padding = 50
                            top = max(0, top - padding)
                            bottom = min(image.shape[0], bottom + padding)
                            left = max(0, left - padding)
                            right = min(image.shape[1], right + padding)
                            
                            face_img = image[top:bottom, left:right]
                            save_path = f"known_faces/{person_name}.jpg"
                            cv2.imwrite(save_path, face_img)
                            
                            save_face_data(st.session_state.username, person_name, save_path)
                            
                            st.success(f"Face saved for {person_name}!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("No face detected in the uploaded image")
                except Exception as e:
                    st.error(f"Error processing uploaded image: {str(e)}")
        
        with col2:
            st.markdown("""
            <div class="stCard">
                <h3>Capture Photo</h3>
                <p>Take a photo using your camera</p>
            </div>
            """, unsafe_allow_html=True)
            
            camera_on = st.checkbox("Start Camera")
            
            if camera_on:
                camera = None
                try:
                    camera = initialize_camera()
                    if camera is None:
                        st.error("Could not access camera")
                        st.stop()
                    
                    camera_placeholder = st.empty()
                    capture_button = st.button("📸 Take Picture")
                    
                    while camera_on:
                        ret, frame = camera.read()
                        if not ret or frame is None:
                            st.error("Failed to capture frame")
                            break
                        
                        # Convert to RGB for display
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(rgb_frame)
                        
                        if capture_button:
                            if not person_name:
                                st.warning("Please enter a name before capturing")
                                break
                            
                            # Save the original BGR frame directly
                            temp_path = f"known_faces/temp_{person_name}.jpg"
                            success = cv2.imwrite(temp_path, frame)
                            
                            if not success:
                                st.error("Failed to save image")
                                break
                            
                            # Read the saved image to verify
                            saved_frame = cv2.imread(temp_path)
                            if saved_frame is None:
                                st.error("Failed to verify saved image")
                                break
                            
                            # Convert to RGB for face detection
                            rgb_saved = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_saved, model="hog")
                            
                            if face_locations:
                                # Crop and save face
                                top, right, bottom, left = face_locations[0]
                                padding = 50
                                top = max(0, top - padding)
                                bottom = min(saved_frame.shape[0], bottom + padding)
                                left = max(0, left - padding)
                                right = min(saved_frame.shape[1], right + padding)
                                
                                face_img = saved_frame[top:bottom, left:right]
                                cv2.imwrite(temp_path, face_img)
                                
                                # Show the captured face
                                st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption="Captured Face")
                                
                                # Save face data
                                if save_face_data(st.session_state.username, person_name, temp_path):
                                    st.success("Face captured and saved successfully!")
                                    time.sleep(1)
                                    st.rerun()
                                break
                            else:
                                st.warning("No face detected. Please position yourself properly.")
                                if Path(temp_path).exists():
                                    Path(temp_path).unlink()
                                time.sleep(2)
                                continue
                        
                        time.sleep(0.1)
                        
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
                finally:
                    if camera is not None:
                        camera.release()
                        cv2.destroyAllWindows()
        
        display_stored_faces(st.session_state.username)
    
    with tab2:
        st.markdown("<h2>Live Recognition</h2>", unsafe_allow_html=True)
        
        # Load known faces for current user
        known_face_encodings, known_face_names = load_known_faces(st.session_state.username)
        
        if not known_face_encodings:
            st.warning("No faces registered yet. Please add faces first.")
        else:
            video_col, info_col = st.columns([3, 1])
            
            with video_col:
                st.markdown("""
                <div class="stCard">
                    <h3>Camera Feed</h3>
                </div>
                """, unsafe_allow_html=True)
                run = st.checkbox('Start Recognition', key='start_recognition')
            
            with info_col:
                st.markdown("""
                <div class="stCard">
                    <h3>Detected People</h3>
                </div>
                """, unsafe_allow_html=True)
                
                confidence_threshold = st.slider(
                    "Recognition Sensitivity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6
                )

            if run:
                camera = None
                detected_names = set()
                
                try:
                    camera = initialize_camera()
                    if camera is None:
                        st.error("Could not access camera. Please check permissions.")
                        st.stop()
                    
                    frame_placeholder = video_col.empty()
                    info_placeholder = info_col.empty()
                    
                    while run:
                        ret, frame = camera.read()
                        if not ret:
                            st.error("Failed to capture frame")
                            break
                        
                        # Process frame
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Find faces in frame
                        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        # Process each face
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            # Compare with known faces
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                            name = "Unknown"
                            confidence = 0
                            
                            if True in matches:
                                first_match_index = matches.index(True)
                                name = known_face_names[first_match_index]
                                
                                # Calculate confidence
                                face_distances = face_recognition.face_distance([known_face_encodings[first_match_index]], face_encoding)
                                confidence = 1 - face_distances[0]
                                
                                if confidence >= confidence_threshold:
                                    detected_names.add(name)
                                    record_attendance(st.session_state.username, name)
                            
                            # Draw box and label
                            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.rectangle(rgb_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(rgb_frame, f"{name} ({confidence:.2%})", 
                                      (left + 6, bottom - 6), 
                                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                        
                        # Update display
                        frame_placeholder.image(rgb_frame)
                        
                        # Update detected names
                        info_placeholder.subheader("Detected People")
                        if detected_names:
                            for name in sorted(detected_names):
                                info_placeholder.write(f"✓ {name}")
                        else:
                            info_placeholder.write("No one detected yet")
                        
                        time.sleep(0.1)
                        
                except Exception as e:
                    st.error(f"Recognition error: {str(e)}")
                finally:
                    if camera is not None:
                        camera.release()
                        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# At the start of the app
def initialize_directories():
    """Create necessary directories if they don't exist"""
    directories = ['known_faces', 'data', 'attendance']
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        
        # Create default users.json if it doesn't exist
        if dir_name == 'data':
            users_file = dir_path / 'users.json'
            if not users_file.exists():
                users_file.write_text('{"users": []}')

# Call this at startup
initialize_directories() 