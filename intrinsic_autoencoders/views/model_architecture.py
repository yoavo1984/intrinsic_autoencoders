"""
This view help with setting up the model architecture.
"""

import numpy as np
import streamlit as st

from app_state import AppState, ModelParameters


class ModelArchitecture:
    def __init__(self):
        self.encoder_number_of_layers: int = 0
        self.decoder_number_of_layers: int = 0
        self.encoder_layers: np.ndarray = np.array([])
        self.decoder_layers: np.ndarray = np.array([])
        self.mirror_encoder: bool = True
        self.app_state: AppState = 0

    def view(self):
        st.title('Model Architecture')

        st.header('Encoder')
        encoder_number_of_layers = st.slider(label='Number of Encoder Layers', min_value=1)
        st.write('---')

        self.encoder_layers = np.zeros(encoder_number_of_layers, dtype=np.uint)
        for i in range(encoder_number_of_layers):
            _, col = st.beta_columns((1, 10))
            self.encoder_layers[i] = col.slider(f'Encoder Layer - {i+1}', value=16, min_value=1, max_value=64)

        st.header('Decoder')
        self.mirror_encoder = st.checkbox(label='Mirror Encoder?')

        if not self.mirror_encoder:
            decoder_number_of_layers = st.slider(label='Number of Decoder Layers', min_value=1)
            st.write('---')

            self.decoder_layers = np.zeros(decoder_number_of_layers, )
            for i in range(decoder_number_of_layers):
                _, col = st.beta_columns((1, 10))
                col.slider(f'Decoder Layer - {i}', value=16, min_value=1, max_value=64)

        self.build_model()

    def update_state(self):
        model_parameters = ModelParameters(encoder_layers=self.encoder_layers,
                                           decoder_layers=self.decoder_layers,
                                           mirror_encoder=self.mirror_encoder)

        self.app_state.model_parameters = model_parameters


