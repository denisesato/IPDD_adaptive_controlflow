from compare_models.controlflow_metric_info import ControlFlowMetricInfo
from metric import Metric


class ControlFlowMetric(Metric):
    def __init__(self, window, trace, metric_name, model1, model2):
        super().__init__(window, metric_name)
        self.diff_added = set()
        self.diff_removed = set()
        self.model1 = model1
        self.model2 = model2

        self.initial_trace = trace
        self.metric_info = ControlFlowMetricInfo(window, trace, metric_name)

    def is_dissimilar(self):
        pass

    def run(self):
        value, diff_added, diff_removed = self.calculate()
        self.metric_info.set_value(value)
        self.metric_info.set_diff_added(diff_added)
        self.metric_info.set_diff_removed(diff_removed)
        self.metric_info.set_dissimilar(self.is_dissimilar())
        self.save_metrics()
