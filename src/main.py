from yolov5.detect_class import predict
import streamlit as st
import os
from PIL import Image
import shutil

st.title("Object Detection for Aksara")

mode = st.selectbox("Pilih aksara", ["jawa"])
upload_file = st.file_uploader("Pilih gambar", type=["png", "jpg"])
mulai = st.button("Prediksi")

if mulai and upload_file:
    
    path_this = os.path.realpath(os.path.dirname(__file__))
    dir_img = os.path.join(path_this, "uploads")
    
    if(os.path.exists(dir_img)):
        shutil.rmtree(dir_img)
        os.mkdir(dir_img)
    else:
        os.mkdir(dir_img)
        
    with open(os.path.join(path_this, "uploads", upload_file.name),"wb") as f:
        f.write(upload_file.read())
    st.header("Gambar awal")
    st.image(upload_file)
    predict(os.path.join(path_this, 'prediction', upload_file.name), "jawa")
    st.header("Gambar di prediksi")
    image = Image.open(os.path.join(path_this, 'prediction', upload_file.name))
    st.image(image)