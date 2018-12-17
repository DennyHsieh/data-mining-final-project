from enum import Enum


class Algorithm(Enum):
    DL_NN_Classifier_r07525035 = {
        "algo_name": "DL_NN_Classifier_r07525035",
        "project_type": "classification"
    }

    @staticmethod
    def get_project_type(algo_name):
        for algo in Algorithm:
            if algo.value['algo_name'] == algo_name:
                return algo.value['project_type']
        return None
