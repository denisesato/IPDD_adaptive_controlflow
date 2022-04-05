import os

from compare_models.compare_dfg import DfgEdgesSimilarityMetric, DfgEditDistanceMetric, \
    DfgNodesSimilarityMetric
from enum import Enum


class Metric(str, Enum):
    NODES = 'Nodes'
    EDGES = 'Edges'


class DfgDefinitions:
    def __init__(self):
        self.models_path = 'dfg'
        self.current_parameters = None
        self.metrics = None

    def set_current_parameters(self, current_parameters):
        self.current_parameters = current_parameters
        self.metrics = current_parameters.metrics

    def get_implemented_metrics(self):
        return Metric

    def get_default_metrics(self):
        return [Metric.NODES]

    def get_model_filename(self, log_name, window):
        map_file = f'{self.models_path}_w{window}.gv'
        return map_file

    def get_metrics_filename(self, current_parameters, metric_name):
        filename = f'{metric_name}_sp{current_parameters.stable_period}.txt'
        return filename

    def get_metrics_path(self, generic_metrics_path, original_filename):
        path = os.path.join(generic_metrics_path, self.models_path, original_filename)
        return path

    def get_models_path(self, generic_models_path, original_filename, activity):
        dfg_models_path = os.path.join(generic_models_path, self.models_path, original_filename,
                                           f'sp{self.current_parameters.stable_period}')
        return dfg_models_path

    def get_metrics_list(self):
        return self.metrics

    def metrics_factory(self, metric_name, window, initial_trace, name, m1, m2, l1, l2, parameters):
        # define todas as métricas existentes para o tipo de modelo de processo
        # porém só serão calculadas as escolhidas pelo usuário (definidas em self.metrics)
        classes = {
            Metric.EDGES.value: DfgEdgesSimilarityMetric(window, initial_trace, name, m1, m2),
            Metric.NODES.value: DfgNodesSimilarityMetric(window, initial_trace, name, m1, m2),
        }
        return classes[metric_name]
