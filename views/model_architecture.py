"""
This view help with setting up the model architecture.
"""

import streamlit as st


class ModelArchitecture:
    def view(self):

        st.title('Model Architecture')

        st.header('Encoder')
        self.number_of_layers = st.slider(label='Number of Encoder Layers', min_value=1)
        st.write('---')
        for i in range(self.number_of_layers):
            _, col = st.beta_columns((1, 10))
            col.slider(f'Encoder Layer - {i}', value=16, min_value=1, max_value=64)

        st.header('Decoder')
        mirror_encoder = st.checkbox(label='Mirror Encoder?')

        if not mirror_encoder:
            self.number_of_layers = st.slider(label='Number of Decoder Layers', min_value=1)
            st.write('---')
            for i in range(self.number_of_layers):
                _, col = st.beta_columns((1, 10))
                col.slider(f'Decoder Layer - {i}', value=16, min_value=1, max_value=64)



ma = ModelArchitecture()
ma.view()