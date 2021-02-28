import time
import importlib

import streamlit as st

import ae_utils
import data_utils

importlib.reload(ae_utils)
importlib.reload(data_utils)


class Training:
    def view(self):
        st.title('Intrinsic AutoEncoders')

        data_f = {
            'regular': data_utils.low_dim_data,
            'transformed': data_utils.low_dim_data_transformed,
            'noisy': data_utils.low_dim_data_with_noise
        }
        st.header('Data Parameters')
        data_type = st.selectbox(label='Transformed Data?', options=list(data_f.keys()))
        power = st.number_input(label='Data To the power', value=1, min_value=1, max_value=10)
        number_of_data_points = st.slider(label='Number of points', min_value=1000, max_value=int(100000), value=10000,
                                          step=1000)
        ambient_dim = st.slider(label='Ambient Dim', min_value=10, max_value=200, value=100)
        intrinsic_dim = st.slider(label='Intrinsic Dim', min_value=10, max_value=200, value=10)

        data = data_f[data_type](ambient_dim=ambient_dim, intrinsic_dim=intrinsic_dim,
                                 num_pts=number_of_data_points)

        data = data ** power
        st.write('Data Shape is :', data.shape)

        st.header('Model Parameters')
        model_intrinsic = st.slider(label='Model Intrinsic Dim', min_value=0, max_value=200, value=10)
        optimizer = st.selectbox(label='Optimizer', options=['adam', 'sgd', 'rmsprop', 'nadam'])
        lr = st.slider(label='Learning Rate', min_value=0., max_value=0.001, value=0.001, step=0.0001)
        st.write(lr)

        st.header('Training Parameters')
        batch_size = st.slider(label='Batch Size', min_value=16, max_value=512, value=64, step=16)
        epochs = st.slider(label='Epochs', min_value=10, max_value=1000, value=10, step=10)

        train = st.button(label='Train')
        multi_train = st.button(label='Multi Train')

        model = ae_utils.create_keras_autoencoder(ambient_dim=ambient_dim, intrinsic_dim=model_intrinsic, optimizer=optimizer,
                                                  lr=lr)


        # model.summary(print_fn=st.write)


        def train_for_100_experiments(model, data):
            import experiment
            importlib.reload(experiment)
            from experiment import AutoEncoderExperiment, AutoEncoderParameters, ExperimentParameters
            from constants import DataType

            def resolve_data_type():
                if not transformed and power == 1:
                    return DataType.LINEAR
                if not transformed and power == 2:
                    return DataType.POWER_2

                return DataType.DEPENDENT

            experiment_params = ExperimentParameters(resolve_data_type())
            st.write('Training Started')
            data = (data - data.mean()) / data.std()
            n_experiments = 10

            n = len(data)
            train_data = data[:int(0.9 * n)]
            test_data = data[int(0.9 * n):]

            for t in [1, 5, 6, 7, 8, 9, 10, 15, 20]:
                for i in range(n_experiments):
                    ae_params = AutoEncoderParameters(t)
                    # We want a new model every time.
                    model = ae_utils.create_keras_autoencoder(ambient_dim=ambient_dim, intrinsic_dim=t,
                                                              optimizer=optimizer,
                                                              lr=lr)
                    ae_utils.train_autoencoder(model, train_data, epochs=30)

                    prediction = model.predict(test_data)

                    results = ((test_data - prediction) ** 2)

                    experiment = AutoEncoderExperiment(experiment_params, ae_params)
                    experiment.set_results(results)
                    experiment.save('results_v3')

            st.write(' --- Done ---')


        # if multi_train:
        #     train_for_100_experiments(model, data)


        if train:
            st.write('Training model... Track on Tensorboard.')
            data = (data - data.mean(axis=0)) / data.std(axis=0)

            samples = data[:5]
            st.write(model.name, model.optimizer._name)

            st.write('sample', samples)
            st.write(model.predict(samples))

            transformed = f'power_{power}_{data_type}'
            ae_utils.train_autoencoder(model, data, batch_size=batch_size, epochs=epochs, data_type=data_type)
            st.write(model.predict(samples))

            for i in range(5):
                print('\a')
                time.sleep(0.2)
