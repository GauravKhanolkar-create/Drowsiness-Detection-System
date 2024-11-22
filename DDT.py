import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
st.set_page_config(page_title="Drowsiness Detection System",page_icon="https://images.emojiterra.com/google/android-12l/512px/1f62a.png",layout="wide")
st.title("DROWSINESS DETECTION SYSTEM")
choice=st.sidebar.selectbox("MENU",("HOME","IP CAMERA","CAMERA"))
if(choice=="HOME"):
    st.sidebar.image("https://t4.ftcdn.net/jpg/01/87/04/55/360_F_187045571_1GPYu7GFoJzKuljKRaeowV20vskm3Hzv.jpg")
    st.image("https://media.licdn.com/dms/image/C4E12AQFLleIL-EGIIQ/article-cover_image-shrink_720_1280/0/1600585924094?e=2147483647&v=beta&t=DcZ2LUJBLCIxzCRnZ1XM0w5dTF7FYToQZxAMUDefoo4")
    st.write("The purpose of the Drowsiness Detection System is to aid in the prevention of accidents passenger and commercial vehicles.")
elif(choice=="IP CAMERA"):
    st.sidebar.image("https://www.dlink.com.sg/wp-content/uploads/2015/04/DCS-7513-mainimage-3-510x600.jpg")
    url=st.text_input("Enter IP CAMERA URL")
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        vid=cv2.VideoCapture(url)
        btn2=st.button("Stop Detection")
        if btn2:
            vid.release()
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier("face.xml")
        eyemodel=load_model("eye.h5",compile=False)
        i=1
        while True:
            flag,frame=vid.read()
            if flag:
                pred=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in pred:
                    face_img=frame[y:y+w,x:x+l]
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                    face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_img=(face_img / 127.5) - 1
                    pred=eyemodel.predict(face_img)[0][0]
                    if(pred>0.9):
                        path="data/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")
elif(choice=="CAMERA"):
    st.sidebar.image("https://images-cdn.ubuy.co.in/633ac165b24f5720bc4f638c-ubuy-online-shopping.jpg")
    cam=st.selectbox("Choose 0 for primary camera or 1 for secondary camera",(0,1))
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        vid=cv2.VideoCapture(cam)
        btn2=st.button("Stop Detection")
        if btn2:
            vid.release()
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier("face.xml")
        eyemodel=load_model("eye.h5",compile=False)
        i=1
        while True:
            flag,frame=vid.read()
            if flag:
                pred=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in pred:
                    face_img=frame[y:y+w,x:x+l]
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                    face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_img=(face_img / 127.5) - 1
                    pred=eyemodel.predict(face_img)[0][0]
                    if(pred>0.9):
                        path="data/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")                        
