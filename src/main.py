import os, sys
path_this = os.path.abspath(os.path.dirname(__file__))
path_root = os.path.abspath(os.path.join(path_this, ".."))
path_yolov5 = os.path.abspath(os.path.join(path_this, "yolov5"))
sys.path.extend([path_this,path_root, path_yolov5])


from yolov5.detect_class import Detection
import streamlit as st
from PIL import Image
import shutil

@st.cache(allow_output_mutation=True)
def load_engine():
    engine = Detection()
    return engine

engine = load_engine()

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
    engine.run(os.path.join(path_this, 'uploads', upload_file.name))
    st.header("Gambar di prediksi")
    image = Image.open(os.path.join(path_this, 'prediction', upload_file.name))
    st.image(image)