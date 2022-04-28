import unittest

from calculate_evaluation_metrics import calculate_metrics_new


class TestMetrics(unittest.TestCase):

    def test_fscore(self):
        real_drifts = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
        detected_drifts = [200, 525, 800, 1102, 1300, 1560, 1825, 2200, 2300, 2450]
        number_of_traces = 2500
        metrics = [
            'f_score',
            'mean_delay',
            'FPR'
        ]
        metric_values = calculate_metrics_new(metrics, detected_drifts, real_drifts, number_of_traces)
        self.assertEqual(round(metric_values['f_score'], 2), 0.8)

        real_drifts = [1000, 2000, 2500, 3500, 4000, 5000, 5500, 6500, 7000, 8000, 8500, 9500, 10000,
                       11000, 11500, 12500, 13000]
        detected_drifts = [1000, 2000, 2500, 3500, 4000, 5000, 5500, 6500, 7000, 8000, 8500, 9500, 10000,
                           11000, 11500, 12500, 13000]
        number_of_traces = 15000
        metric_values = calculate_metrics_new(metrics, detected_drifts, real_drifts, number_of_traces)
        self.assertEqual(round(metric_values['f_score'], 2), 1)


if __name__ == '__main__':
    unittest.main()
