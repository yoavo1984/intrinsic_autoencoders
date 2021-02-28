from importlib import reload
import streamlit as st


import views.training
import views.model_architecture
import views.dataset_synthesizer
import views.playground
import views
reload(views.training);reload(views.model_architecture);reload(views.dataset_synthesizer);reload(views.playground)


from views.training import Training
from views.model_architecture import ModelArchitecture
from views.dataset_synthesizer import DatasetSynthesizer
from views.playground import Playground

views = {
    'Training': Training(),
    'ModelArchitecture': ModelArchitecture(),
    'DatasetSynthesizer': DatasetSynthesizer(),
    'Playground': Playground()
}

if __name__ == '__main__':
    stb = st.sidebar

    stb.image('res/idc_logo.png', width=250)
    page = stb.selectbox(label='Page', options=list(views.keys()))

    views[page].view()