from json_tricks import dumps
from metric_info import MetricInfo, AdditionalInfo


class ControlFlowMetricInfo(MetricInfo):
    def __init__(self, window, trace, metric_name):
        super().__init__(window, trace, metric_name)

    def set_diff_added(self, diff):
        if len(diff) > 0:
            self.add_additional_info(AdditionalInfo('Added', diff))

    def set_diff_removed(self, diff):
        if len(diff) > 0:
            self.add_additional_info(AdditionalInfo('Removed', diff))
