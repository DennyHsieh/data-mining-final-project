from abc import ABC
from abc import abstractmethod
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinition:
    def __init__(self, name, type, range, default_value, description):
        self.name = name
        self.type = type
        self.range = range
        self.default_value = default_value
        self.description = description

    def get_params_definition(self):
        return self.__dict__


class ParamsDefinitionSet:
    def __init__(self):
        self.params_definition_set = {}
        raise NotImplementedError

    def get_params_definition_json_list(self):
        definition_set_json_list = []
        for params_object in self.params_definition_set:
            definition_set_json_list.append(params_object.get_params_definition())
        return definition_set_json_list


# Inanalysis Algorithm interface
class InanalysisAlgo(ABC):
    @abstractmethod
    def do_algo(self, input_params):
        raise NotImplementedError

    def check_input_params(self, input_params_definition, user_input_params):
        try:
            if len(user_input_params) != len(input_params_definition):
                log.error("Length of user_input_params is not equal to input_params_definition")
                return False
            for params_dict in input_params_definition:
                params_name = params_dict['name']
                user_input_value = user_input_params[params_name]
                if user_input_value != params_dict['default_value'] and user_input_value is not None:
                    if params_dict['type'] is 'int':
                        if type(user_input_value) is not int:
                            log.error(str(user_input_value) + " is not int")
                            return False
                        if params_dict['range'].find(',') is not -1:
                            l = list(map(float, params_dict['range'].split(',')))
                            if user_input_value < l[0] or user_input_value > l[1]:
                                log.error(str(user_input_value) + " is out of range (" + l[0] + "~" + l[1] + ")")
                                return False
                    elif params_dict['type'] is 'float':
                        if type(user_input_value) is not float:
                            log.error(str(user_input_value) + " is not float")
                            return False
                        if params_dict['range'].find(',') is not -1:
                            l = list(map(float, params_dict['range'].split(',')))
                            if user_input_value < l[0] or user_input_value > l[1]:
                                log.error(str(user_input_value) + " is out of range (" + l[0] + "~" + l[1] + ")")
                                return False
                    elif params_dict['type'] is 'boolean':
                        if type(user_input_value) is not bool:
                            log.error(str(user_input_value) + " is not bool")
                            return False
                    elif params_dict['type'] is 'enum':
                        if user_input_value not in params_dict['range'].split(','):
                            log.error(str(user_input_value) + " is not in " + params_dict['range'])
                            return False
        except Exception as e:
            log.error(str(e))
            return False
        return True


# 控制 model 的參數設定
class AlgoControl:
    def __init__(self, input_params):
        self.control_params = input_params['control_params']
        self.mode = input_params['mode']


# 用於接收 data
class AlgoData:
    def __init__(self, input_params):
        self.data = input_params['data']
        self.label = input_params['label']


# model 本身所產生的訓練參數
class AlgoModel:
    def __init__(self, input_params):
        self.model_instance = input_params['model_instance']
        self.model_params = input_params['model_params']


# 統一收集參數
class AlgoParam:
    def __init__(self, algo_control, algo_data, algo_model):
        self.algo_control = AlgoControl(algo_control)
        self.algo_data = AlgoData(algo_data)
        self.algo_model = AlgoModel(algo_model)
