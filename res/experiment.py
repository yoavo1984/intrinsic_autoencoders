"""
The expermient class provides an easy and reporoducible way to interact with experiments.
"""
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import NamedTuple, Tuple, Generator

import numpy as np

import constants


class Experiment(ABC):
    @abstractmethod
    def save(self, experiment_path: str) -> None:
        """
        Save the experiment to disk.
        """
        pass


AutoEncoderParameters = NamedTuple('AutoEncoderParameters', [('bottleneck', int)])
ExperimentParameters = NamedTuple('ExperimentParameters', [('data_type', constants.DataType)])


def build_experiment_directory_path(experiment_path: str, ae_params: AutoEncoderParameters,
                                    experiment_params: ExperimentParameters):
    experiment_path = os.path.join(experiment_path,
                                   experiment_params.data_type,
                                   str(ae_params.bottleneck))

    return experiment_path


class AutoEncoderExperiment(Experiment):
    """
    Contains the relevant parameter to describe an autoencoder training experiment.
    """

    def __init__(self, experiment_params: ExperimentParameters, ae_params: AutoEncoderParameters):
        self.experiment_params = experiment_params
        self.ae_params = ae_params
        self.results = np.array([])

    def set_results(self, results: np.ndarray):
        self.results = results

    def get_results_mean(self):
        return self.results.mean()

    def get_results_std(self):
        return self.results.std()

    def save(self, experiment_path: str) -> None:
        now_str = datetime.now().strftime('%m-%d-%H-%M-%S')
        experiment_path = build_experiment_directory_path(experiment_path, self.ae_params, self.experiment_params)

        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        results_path = os.path.join(experiment_path, now_str)
        np.save(results_path, self.results)


class AutoEncoderExperimentBuilder:
    """
    A class for constructing AutoEncoderExperiment objects.
    """

    @staticmethod
    def build_experiment(path: str, ae_params: AutoEncoderParameters, experiment_params: ExperimentParameters):
        experiment = AutoEncoderExperiment(experiment_params, ae_params)

        results = np.load(path)
        experiment.set_results(results)

        return experiment

    @staticmethod
    def get_all_experiments(experiment_path: str,
                            experiment_params: ExperimentParameters,
                            ae_params: AutoEncoderParameters) -> Generator[AutoEncoderExperiment, None, None]:
        """
        Get all the experiment that adhere to a certain parameters setting.
        """
        experiments_path = build_experiment_directory_path(experiment_path, ae_params, experiment_params)

        files = os.listdir(experiments_path)
        files = [os.path.join(experiments_path, f) for f in files]

        for f in files:
            experiment = AutoEncoderExperimentBuilder.build_experiment(f, ae_params, experiment_params)
            yield experiment

    @staticmethod
    def get_all_experiments_statistics(experiment_path: str,
                                       experiment_params: ExperimentParameters,
                                       ae_params: AutoEncoderParameters, ) -> Tuple[float, float]:
        experiments = AutoEncoderExperimentBuilder.get_all_experiments(experiment_path, experiment_params, ae_params)

        mses = []
        for experiment in experiments:
            mses.append(experiment.get_results_mean())

        mses = np.array(mses)

        return mses.mean(), mses.std()


if __name__ == '__main__':
    aeeb = AutoEncoderExperimentBuilder()

    print(aeeb.get_all_experiments_statistics('results_v3', ExperimentParameters(constants.DataType.LINEAR),
                                              AutoEncoderParameters(1)))
