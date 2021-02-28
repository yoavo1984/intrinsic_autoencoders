import streamlit as st


class DatasetSynthesizer:
    def view(self):
        data_type = st.selectbox(label='Which type of data would you like to synthesize?', options=['1', '2', '3'])
        self.tunnel_to_sub_view(data_type)

    def tunnel_to_sub_view(self, data_type):
        if data_type == '1':
            self.sub_view_1()

        if data_type == '1':
            pass

        if data_type == '1':
            pass

    def sub_view_1(self):
        pass
