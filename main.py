from importlib.resources import path
from fastai.vision.all import *
import streamlit as st
import plotly.express as px
import pathlib

#Fixing path
temp = pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

#Info about project
st.title('Obyektlarni klassifikatsiya qiluvchi model')
st.header(f'"Browse files" tugmasini bosib, yoki telefoningizda: Mashina, Samolet, Vertolet, Velosiped, Qayiq, Daraxt va Qurol rasmlaridan birini kiriting va sizga qaysi klassga kirishini bashorat qilib beradi')

#Uploading image
file = st.file_uploader('Rasm yuklash',type=['png','jpg','jpeg','gif'])
st.info('Rasm formati <"png" "jpg" "jpeg" "gif"> formatlarida bolishi kerak ')
if file is None:
    file = st.camera_input("Yoki obyektni rasmga oling")

#Agar Fayl mavjud bolsa keyin bularni ishlatamiz
if file:
    st.image(file)
    #PIL Convert
    img = PILImage.create(file)

    #modelni yuklaymiz
    model = load_learner('classificator.pkl')

    #Bahorat qilish
    pred, pred_id, probs  = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}')

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
