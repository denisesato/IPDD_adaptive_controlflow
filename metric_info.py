from json_tricks import dumps


class MetricInfo:
    def __init__(self, window, trace, metric_name):
        self.value = 0
        self.window = window
        self.initial_trace = trace
        self.metric_name = metric_name
        self.dissimilar = False
        self.info = []
        # the complete info shows the p-value calculated for each activity
        # when statistical hypothesis test is applied (Sojourn Time Similarity)
        self.complete_info = None

    def add_additional_info(self, additional_info):
        self.info.append(additional_info)

    def is_dissimilar(self):
        return self.dissimilar

    def serialize(self):
        result = dumps(self)
        return result

    def set_value(self, value):
        self.value = value

    def set_dissimilar(self, dissimilar):
        self.dissimilar = dissimilar

    def get_additional_info(self):
        return self.info

    def include_complete_info(self, complete_info):
        self.complete_info = complete_info

    def get_complete_info(self):
        return self.complete_info

    def __str__(self):
        info = f'Metric name: {self.metric_name}\n'
        info += f'Window: {self.window}\n'
        info += f'Value: {self.value}\n'
        info += f'Is dissimilar: {self.dissimilar}\n'
        info += f'Additional info: {[info.get_status_info() for info in self.info]}\n'
        if self.complete_info:
            info += f'Complete info: {self.complete_info}'
        return info


class AdditionalInfo:
    def __init__(self, name, information_set):
        self.name = name
        self.information_set = information_set

    def get_status_info(self):
        strType = f'{self.name}: '
        strList = ''
        for info in self.information_set:
            if len(info) == 3:  # edges
                strList += f'[{info[0]} - {info[1]}] '
            else:  # nodes
                strList += f'[{info}] '
        return strType, strList