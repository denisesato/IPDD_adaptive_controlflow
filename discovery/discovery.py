import os

from graphviz import Source


class Discovery:
    def generate_process_model(self, sub_log, models_path, event_data_original_name, w_count, activity=''):
        pass

    def get_process_model(self, models_path, log_name, window, activity):
        map_file = self.model_type_definitions.get_model_filename(log_name, window)
        models_path = self.model_type_definitions.get_models_path(models_path, log_name, activity)

        if os.path.exists(os.path.join(models_path, map_file)):
            gviz = Source.from_file(filename=map_file, directory=models_path)
            return gviz.source

        return """
                digraph  {
                  node[style="filled"]
                  a ->b->d
                  a->c->d
                }
                """