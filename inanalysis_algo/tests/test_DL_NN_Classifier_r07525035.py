import unittest
import inanalysis_algo.algo_component as alc
from inanalysis_algo.utils import AlgoUtils
from inanalysis_algo.utils import Algorithm
from keras.datasets import mnist
from sklearn.datasets import load_iris, load_boston
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class InAlgoTestCase(unittest.TestCase):

    def setUp(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # data pre-processing
        self.X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
        self.X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
        self.y_train = y_train
        # self.y_test = y_test

        iris = load_iris()
        self.iris_data = iris.data
        self.iris_label = iris.target

        boston = load_boston()
        self.boston_data = boston.data
        self.boston_label = boston.target

    def tearDown(self):
        del self.X_train
        del self.X_test
        del self.y_train
        # del self.y_test

        del self.iris_data
        del self.iris_label
        del self.boston_data
        del self.boston_label

    def test_correct_dl_nn_classifier_parameter_type(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 10,
            "layer_3_activation": "softmax",
            "optimizer": "RMSprop",
            "loss": "categorical_crossentropy",
            "epochs": 3,
            "batch_size": 32
        }
        algo_name = 'DL_NN_Classifier_r07525035'
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.X_train, 'label': self.y_train},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory(algo_name)
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is True)
        self.assertEqual(Algorithm.get_project_type(algo_name), "classification")

    def test_error_dl_nn_classifier_parameter_neuron_1_string_type(self):
        # given: error input parameter "layer_1_neuron" needs to be type(int)
        arg_dict = {
            "layer_1_neuron": "relu",
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 10,
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": 2,
            "batch_size": 32,
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_activation_1_int_type(self):
        # given: error input parameter "layer_1_activation" needs to be type(enum)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": 24,
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 8,
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": 3,
            "batch_size": 32
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_neuron_2_string_type(self):
        # given: error input parameter "layer_2_neuron" needs to be type(int)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": "relu",
            "layer_2_activation": "tanh",
            "layer_3_neuron": 10,
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": 2,
            "batch_size": 32,
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_activation_2_int_type(self):
        # given: error input parameter "layer_2_activation" needs to be type(enum)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": 12,
            "layer_3_neuron": 8,
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": 3,
            "batch_size": 32
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_neuron_3_string_type(self):
        # given: error input parameter "layer_3_neuron" needs to be type(int)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": "relu",
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": 2,
            "batch_size": 32,
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_activation_3_int_type(self):
        # given: error input parameter "layer_3_activation" needs to be type(eum)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 8,
            "layer_3_activation": 4,
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": 3,
            "batch_size": 32
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_optimizer_int_type(self):
        # given: error input parameter "optimizer" needs to be type(enum)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 8,
            "layer_3_activation": "softmax",
            "optimizer": 4,
            "loss": "mean_squared_error",
            "epochs": 3,
            "batch_size": 32
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_loss_int_type(self):
        # given: error input parameter "loss" needs to be type(enum)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 8,
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": 4,
            "epochs": 3,
            "batch_size": 32
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_epoch_string_type(self):
        # given: error input parameter "epochs" needs to be type(int)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 8,
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": "relu",
            "batch_size": 32
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_error_dl_nn_classifier_parameter_batch_size_string_type(self):
        # given: error input parameter "batch_size" needs to be type(int)
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 8,
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": 3,
            "batch_size": "relu"
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        input_params_definition = in_algo.get_input_params_definition()
        # when: checkout input type
        check_result = in_algo.check_input_params(input_params_definition, algo_input.algo_control.control_params)
        # then: type match
        self.assertTrue(check_result is False)

    def test_correct_dl_nn_classifier_do_algo_iris_dataset(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "layer_1_neuron": 8,
            "layer_1_activation": "relu",
            "layer_2_neuron": 4,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 3,
            "layer_3_activation": "softmax",
            "optimizer": "Adam",
            "loss": "categorical_crossentropy",
            "epochs": 200,
            "batch_size": 5
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.iris_data, 'label': self.iris_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        log.debug(algo_input)
        # when: do decision tree algorithm
        algo_output = in_algo.do_algo(algo_input)

        # then:
        self.assertTrue(algo_output is not None)
        self.assertTrue(algo_output.algo_model.model_instance is not None)

    def test_correct_dl_nn_classifier_do_algo_mnist_dataset(self):
        # given: collect input parameter, create algorithm object
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "tanh",
            "layer_3_neuron": 10,
            "layer_3_activation": "softmax",
            "optimizer": "RMSprop",
            "loss": "categorical_crossentropy",
            "epochs": 3,
            "batch_size": 32
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.X_train, 'label': self.y_train},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        log.debug(algo_input)
        # when: do decision tree algorithm
        algo_output = in_algo.do_algo(algo_input)

        # then:
        self.assertTrue(algo_output is not None)
        self.assertTrue(algo_output.algo_model.model_instance is not None)

    def test_error_dl_nn_classifier_do_algo_wrong_dataset(self):
        # given: error database input, create algorithm object
        arg_dict = {
            "layer_1_neuron": 32,
            "layer_1_activation": "relu",
            "layer_2_neuron": 16,
            "layer_2_activation": "relu",
            "layer_3_neuron": 10,
            "layer_3_activation": "relu",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "epochs": 3,
            "batch_size": 32
        }
        algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                                   algo_data={'data': self.boston_data, 'label': self.boston_label},
                                   algo_model={'model_params': None, 'model_instance': None})
        in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
        log.debug(algo_input)
        # when: do decision tree algorithm
        try:
            algo_output = in_algo.do_algo(algo_input)
        except:
            algo_output = None

        # then:
        self.assertTrue(algo_output is None)


if __name__ == '__main__':
    unittest.main()
