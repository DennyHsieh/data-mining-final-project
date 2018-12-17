import inanalysis_algo.algo_component as alc
import logging
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
import keras

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ParamsDefinitionSet(alc.ParamsDefinitionSet):
    def __init__(self):
        self.params_definition_set = \
            {
                alc.ParamsDefinition(name='layer_1_neuron', type='int', range='1, 512', default_value='32',
                                     description=''),
                alc.ParamsDefinition(name='layer_1_activation', type='enum', range='softmax,relu,tanh,sigmoid,linear',
                                     default_value='relu', description=''),
                alc.ParamsDefinition(name='layer_2_neuron', type='int', range='1, 512', default_value='16',
                                     description=''),
                alc.ParamsDefinition(name='layer_2_activation', type='enum', range='softmax,relu,tanh,sigmoid,linear',
                                     default_value='tanh', description=''),
                alc.ParamsDefinition(name='layer_3_neuron', type='int', range='1, 512', default_value='8',
                                     description=''),
                alc.ParamsDefinition(name='layer_3_activation', type='enum', range='softmax,relu,tanh,sigmoid,linear',
                                     default_value='softmax', description=''),
                alc.ParamsDefinition(name='optimizer', type='enum', range='SGD,RMSprop,Adagrad,Adadelta,Adam',
                                     default_value='Adam', description=''),
                alc.ParamsDefinition(name='loss', type='enum',
                                     range='mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,'
                                           'mean_squared_logarithmic_error,squared_hinge,hinge,categorical_hinge,'
                                           'categorical_crossentropy,logcosh',
                                     default_value='mean_squared_error', description=''),
                alc.ParamsDefinition(name='epochs', type='int', range='', default_value='3', description=''),
                alc.ParamsDefinition(name='batch_size', type='int', range='', default_value='32', description=''),
            }


class NN_Classifier(alc.InanalysisAlgo):
    def __init__(self):
        self.input_params_definition = ParamsDefinitionSet()

    def get_input_params_definition(self):
        return self.input_params_definition.get_params_definition_json_list()

    def do_algo(self, input):
        control_params = input.algo_control.control_params
        if not self.check_input_params(self.get_input_params_definition(), control_params):
            log.error("Check input params type error.")
            return None
        mode = input.algo_control.mode
        data = input.algo_data.data
        label = np_utils.to_categorical(input.algo_data.label, num_classes=control_params['layer_3_neuron'])

        def baseline_model():
            keras.backend.clear_session()
            model = Sequential()
            model.add(Dense(control_params['layer_1_neuron'], input_dim=data.shape[1]))
            model.add(Activation(control_params['layer_1_activation']))
            model.add(Dense(control_params['layer_2_neuron']))
            model.add(Activation(control_params['layer_2_activation']))
            model.add(Dense(control_params['layer_3_neuron']))
            model.add(Activation(control_params['layer_3_activation']))

            # output layer
            model.compile(optimizer=control_params['optimizer'], loss=control_params['loss'], metrics=['accuracy'])

            return model

        if mode == 'training':
            try:
                model = KerasClassifier(build_fn=baseline_model, epochs=control_params['epochs'],
                                        batch_size=control_params['batch_size'], verbose=False)
                model.fit(data, label)

                # model = baseline_model()
                # model.fit(data, label, epochs=control_params['nb_epoch'], batch_size=control_params['batch_size'])

                algo_output = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': ''},
                                            algo_data={'data': data, 'label': label},
                                            algo_model={'model_params': model.get_params(), 'model_instance': model})
            except Exception as e:
                log.error(str(e))
                algo_output = None
        else:
            algo_output = None
        return algo_output
