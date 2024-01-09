import streamlit as st
from fastai.vision.all import *
from PIL import Image

# title
st.title('Transport classification model')

# upload
file = st.file_uploader('Picture upload', type=['png', 'jpeg', 'gif', 'svg'])

if file is not None:
    img = PILImage.create(file)

    # model
    model = load_learner('transport_model.pkl', pickle_module=pickle)

    # prediction
    prediction = model.predict(img)
    
    # display prediction
    st.success(prediction)
import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title('Transport classification model')


# upload
file = st.file_uploader('Upload picture', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)
    # model
    model = load_learner('transport_model.pkl')

    # prediction
    pred, pred_id, probs=model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)