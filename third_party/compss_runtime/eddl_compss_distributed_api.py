import io
#from pyeddl import _core
import pyeddl._core.eddl as eddl
from eddl_compss_distributed import *
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on

import numpy as np

from net_utils import net_aggregateParameters
from net_utils import net_parametersToNumpy
from net_utils import net_numpyParametersFill
import os

compss_object = None

def build(model, optimizer, losses, metrics, compserv, random_weights):

    # Initialize the compss object
    global compss_object
    compss_object = Eddl_Compss_Distributed()

    # This call to build is for security reasons to initialize the parameters to random values
    eddl.build(
        model,
        optimizer,
        losses,
        metrics,
        compserv,
        random_weights
    )

    # Serialize the model so it can be sent through the network
    serialized_model = eddl.serialize_net_to_onnx_string(model, False)

    # Build the model in all the nodes synchronously
    compss_object.build(serialized_model, "", losses, metrics, "")


def train_batch(model, x_train, y_train, num_workers, num_epochs_for_param_sync, workers_batch_size):

    global compss_object

    # Initial parameters that every node will take in order to begin the training
    initial_parameters = net_parametersToNumpy(model.getParameters())
    recv_weights = [list() for i in range(0, num_workers)]

    s = eddlT.getShape(x_train)
    num_images_per_worker = int(s[0] / num_workers)

    print("Num workers: ", num_workers)
    print("Num images per worker: ", num_images_per_worker)
    print("Workers batch size: ", workers_batch_size)

    for i in range(0, num_workers):
        recv_weights[i] = compss_object.train_batch(initial_parameters, num_images_per_worker, num_epochs_for_param_sync, workers_batch_size)

    # COMPSS barrier to force waiting until every node finishes its training (synchronous training)
    recv_weights = compss_wait_on(recv_weights)

    # Parameters aggregation
    final_weights = net_aggregateParameters(recv_weights)

    # Set trained and aggregated parameters to the model
    model.setParameters(net_parametersToTensor(final_weights))


def train_batch_async(model, x_train, y_train, num_workers, num_epochs_for_param_sync, workers_batch_size):

    global compss_object

    s = eddlT.getShape(x_train)
    num_images_per_worker = int(s[0] / num_workers)

    print("Num workers: ", num_workers)
    print("Num images per worker: ", num_images_per_worker)
    print("Workers batch size: ", workers_batch_size)

    # Array of final parameters whose initial value is initial parameters
    accumulated_parameters = net_parametersToNumpy(model.getParameters())
    #accumulated_parameters = net_numpyParametersFill(accumulated_parameters, 0)

    workers_parameters = [net_parametersToNumpy(model.getParameters()) for i in range(0, num_workers)]

    for j in range(0, num_workers):

        workers_parameters[j] = compss_object.train_batch(workers_parameters[j], num_images_per_worker, num_epochs_for_param_sync, workers_batch_size)
        workers_parameters[j] = compss_object.aggregate_parameters_async(accumulated_parameters, workers_parameters[j], 1 / num_workers)


    accumulated_parameters = compss_wait_on(accumulated_parameters)
    #workers_parameters = compss_wait_on(workers_parameters)

    print("Workers parameters: ", workers_parameters)
    print("Final accumulated parameters: ", accumulated_parameters)

    # Set trained and aggregated parameters to the model
    model.setParameters(net_parametersToTensor(accumulated_parameters))


def fit_async(model, x_train, y_train, num_workers, num_epochs_for_param_sync, max_num_async_epochs, workers_batch_size):

    global compss_object

    s = eddlT.getShape(x_train)
    num_images_per_worker = int(s[0] / num_workers)

    print("Num workers: ", num_workers)
    print("Num images per worker: ", num_images_per_worker)
    print("Workers batch size: ", workers_batch_size)

    # Array of final parameters whose initial value is initial parameters
    accumulated_parameters = net_parametersToNumpy(model.getParameters())
    #accumulated_parameters = net_numpyParametersFill(accumulated_parameters, 0)

    workers_parameters = [net_parametersToNumpy(model.getParameters()) for i in range(0, num_workers)]

    for i in range(0, max_num_async_epochs):
        for j in range(0, num_workers):

            workers_parameters[j] = compss_object.train_batch(workers_parameters[j], num_images_per_worker, num_epochs_for_param_sync, workers_batch_size)
            workers_parameters[j] = compss_object.aggregate_parameters_async(accumulated_parameters, workers_parameters[j], 1 / num_workers)


    accumulated_parameters = compss_wait_on(accumulated_parameters)
    #workers_parameters = compss_wait_on(workers_parameters)

    print("Workers parameters: ", workers_parameters)
    print("Final accumulated parameters: ", accumulated_parameters)

    # Set trained and aggregated parameters to the model
    model.setParameters(net_parametersToTensor(accumulated_parameters))