from setuptools import setup, find_packages

setup(
    name="facial_recognition",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.28.0",
        "opencv-python-headless==4.8.1.78",
        "face-recognition==1.3.0",
        "numpy==1.26.3",
        "Pillow==10.0.0",
        "dlib-binary==19.24.0",
        "setuptools>=69.0.0",
    ],
    python_requires=">=3.12",
) 