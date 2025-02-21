import json
import hashlib
from pathlib import Path
import streamlit as st
import time
import cv2
import os



class Auth:
    def __init__(self):
        self.users_file = Path("data/users.json")
        self.users_file.parent.mkdir(exist_ok=True)
        if not self.users_file.exists():
            self.users_file.write_text('{"users": []}')
        
    def load_users(self):
        return json.loads(self.users_file.read_text())
    
    def save_users(self, users_data):
        self.users_file.write_text(json.dumps(users_data, indent=4))
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def signup(self, username, password):
        users_data = self.load_users()
        
        if any(user['username'] == username for user in users_data['users']):
            return False, "Username already exists"
        
        # Create user-specific directory
        user_dir = Path(f"known_faces/{username}")
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Add new user
        users_data['users'].append({
            'username': username,
            'password': self.hash_password(password),
            'known_faces': [],
            'attendance_records': []
        })
        
        self.save_users(users_data)
        return True, "Signup successful"
    
    def login(self, username, password):
        users_data = self.load_users()
        
        # Find user
        user = next((user for user in users_data['users'] 
                    if user['username'] == username 
                    and user['password'] == self.hash_password(password)), None)
        
        if user:
            return True, "Login successful"
        return False, "Invalid username or password"

def show_auth_page():
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        border: 1px solid rgba(0, 255, 157, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Input styling */
    .stTextInput input, .stTextInput textarea {
        color: black !important;
        background-color: white !important;
    }
    
    .auth-title {
        text-align: center;
        color: #00ff9d;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
    }
    
    .auth-switch {
        text-align: center;
        margin-top: 1rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .loading-animation {
        text-align: center;
        color: #00ff9d;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    auth = Auth()
    
    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = None
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    st.markdown("<h1 class='auth-title'>Face Recognition System</h1>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
        
        if st.session_state.show_signup:
            st.subheader("Sign Up")
            with st.form("signup_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup_button = st.form_submit_button("Sign Up")
                
                if signup_button:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif not new_username or not new_password:
                        st.error("Please fill all fields")
                    else:
                        success, message = auth.signup(new_username, new_password)
                        if success:
                            st.success(message)
                            # Show loading animation
                            with st.spinner("Logging in..."):
                                time.sleep(1)
                            st.session_state.auth_status = True
                            st.session_state.username = new_username
                            st.experimental_rerun()
                        else:
                            st.error(message)
        else:
            st.subheader("Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")
                
                if login_button:
                    if not username or not password:
                        st.error("Please fill all fields")
                    else:
                        success, message = auth.login(username, password)
                        if success:
                            with st.spinner("Logging in..."):
                                time.sleep(1)
                            st.session_state.auth_status = True
                            st.session_state.username = username
                            st.experimental_rerun()
                        else:
                            st.error(message)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Toggle between login and signup
        st.markdown("<div class='auth-switch'>", unsafe_allow_html=True)
        if st.session_state.show_signup:
            if st.button("Already have an account? Login"):
                st.session_state.show_signup = False
                st.experimental_rerun()
        else:
            if st.button("Don't have an account? Sign Up"):
                st.session_state.show_signup = True
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state.auth_status 