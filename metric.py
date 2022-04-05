import threading


class Metric(threading.Thread):
    def __init__(self, window, metric_name):
        super().__init__()
        self.value = 0
        self.window = window
        self.metric_name = metric_name
        self.filename = None
        self.lock = None
        self.manager_similarity_metrics = None

    def set_saving_definitions(self, filename, current_parameters, lock, manager_similarity_metrics):
        self.filename = filename
        self.lock = lock
        self.manager_similarity_metrics = manager_similarity_metrics

    def get_info(self):
        return self.metric_info

    def get_complete_info(self):
        return self.metric_info.get_complete_info()

    def is_dissimilar(self):
        pass

    def calculate(self):
        pass

    def save_metrics(self):
        # save the metric when it is dissimilar or if the metric calculate p-values for each activitu (complete_info)
        if self.is_dissimilar() or self.get_complete_info():
            self.lock.acquire()
            # update the file containing the metrics' values
            with open(self.filename, 'a+') as file:
                #print(f'---------------------- Vai salvar dados sobre métrica \n{str(self.get_info())}')
                file.write(self.get_info().serialize())
                file.write('\n')
            self.lock.release()
            print(f'Saving [{self.metric_name}] comparing windows [{self.window-1}-{self.window}]')
        self.manager_similarity_metrics.increment_metrics_count()
        self.manager_similarity_metrics.check_finish()

    def run(self):
        pass
