import streamlit as st

from views.training import Training
from views.model_architecture import ModelArchitecture
from views.dataset_synthesizer import DatasetSynthesizer

views = {
    'Training': Training(),
    'ModelArchitecture': ModelArchitecture()
}

if __name__ == '__main__':
    stb = st.sidebar

    stb.image('idc_logo.png', width=250)
    page = stb.selectbox(label='Page', options=list(views.keys()))

    views[page].view()