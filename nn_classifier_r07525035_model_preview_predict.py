import matplotlib.pyplot as plt
import inanalysis_algo.algo_component as alc
from inanalysis_algo.utils import AlgoUtils
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd
import numpy as np
import itertools
import os


def dl_nn_model_preview(data, predict_result, x_axis_name, y_axis_name):
    plt.scatter(data[x_axis_name], data[y_axis_name], c=predict_result, edgecolor='black', linewidth='1')
    plt.title("Iris model plot")
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    # plt.savefig(x_axis_name + " _ " + y_axis_name + ".png")
    plt.show()
    plt.close('all')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          fname=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        picname = fname + '_cm_nornalize.png'
    else:
        print('Confusion matrix, without normalization')
        picname = fname + '_cm_without_nornalize.png'

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(fname + title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Generate folder of export files
    export_folder = 'export_result'
    if not os.path.exists(export_folder):
        os.mkdir(export_folder)
    file_folder = os.path.join(export_folder, fname)
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    plt.savefig(os.path.join(file_folder, picname))


def classification_report_csv(report, fname):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split(' ')
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)

    # Generate folder of export files
    export_folder = 'export_result'
    if not os.path.exists(export_folder):
        os.mkdir(export_folder)
    file_folder = os.path.join(export_folder, fname)
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    dataframe.to_csv(os.path.join(file_folder, fname + '_classification_report.csv'), index=False)


def model_predict(arg_dict, data_train, label_train, data_test, label_test, class_names, fname):
    algo_input = alc.AlgoParam(algo_control={'mode': 'training', 'control_params': arg_dict},
                               algo_data={'data': data_train, 'label': label_train},
                               algo_model={'model_params': None, 'model_instance': None})
    in_algo = AlgoUtils.algo_factory('DL_NN_Classifier_r07525035')
    algo_output = in_algo.do_algo(algo_input)
    model = algo_output.algo_model.model_instance

    predict_result = model.predict(data_test)

    clsf_report = classification_report(label_test, predict_result)
    print("Classification report for classifier %s:\n%s\n" % (model, clsf_report))
    classification_report_csv(clsf_report, fname)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(label_test, predict_result)
    np.set_printoptions(precision=2, suppress=True)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, fname=fname,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          fname=fname, title='Normalized confusion matrix')
    plt.show()


if __name__ == "__main__":
    # Iris dataset
    iris_arg_dict = {
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
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_label = iris.target

    iris_data_train, iris_data_test, iris_label_train, iris_label_test = train_test_split(iris_data, iris_label,
                                                                                          test_size=0.5)
    iris_class_names = iris.target_names

    model_predict(iris_arg_dict, iris_data_train, iris_label_train, iris_data_test, iris_label_test, iris_class_names,
                  'iris')

    # Digit dataset
    digits_arg_dict = {
        "layer_1_neuron": 32,
        "layer_1_activation": "relu",
        "layer_2_neuron": 16,
        "layer_2_activation": "relu",
        "layer_3_neuron": 10,
        "layer_3_activation": "softmax",
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "epochs": 10,
        "batch_size": 3
    }

    digits = load_digits()
    digits_data = digits.data
    digits_label = digits.target
    digits_data_train, digits_data_test, digits_label_train, digits_label_test = train_test_split(digits_data,
                                                                                                  digits_label,
                                                                                                  test_size=0.2)
    digits_class_names = np.linspace(0, 9, 10, dtype='int')

    model_predict(digits_arg_dict, digits_data_train, digits_label_train, digits_data_test, digits_label_test,
                  digits_class_names, 'digits')

    # MNIST dataset
    mnist_arg_dict = {
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

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    mnist_class_names = np.linspace(0, 9, 10, dtype='int')

    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize

    model_predict(mnist_arg_dict, X_train, y_train, X_test, y_test, mnist_class_names, 'mnist')
