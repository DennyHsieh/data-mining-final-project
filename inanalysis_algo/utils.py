import logging
from inanalysis_algo.classification.DL_NN_Classifier_r07525035 import NN_Classifier as DL_NN_classifier_r07525035
from inanalysis_algo.algo_enum import Algorithm
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class AlgoUtils:
    @staticmethod
    def algo_factory(model_method):
        model_method_dict = {
            Algorithm.DL_NN_Classifier_r07525035.value.get('algo_name'): DL_NN_classifier_r07525035,
        }
        
        if model_method in model_method_dict.keys():
            algo = model_method_dict[model_method]
            return algo()
        else:
            return None
