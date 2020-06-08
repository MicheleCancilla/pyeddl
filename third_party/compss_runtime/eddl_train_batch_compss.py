"""\
TRAIN_BATCH example.
"""
from os import environ
import platform

import argparse
import sys
from timeit import default_timer as timer

import numpy as np
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import *

from net_utils import net_parametersToNumpy
from net_utils import net_parametersToTensor
from net_utils import getFreqFromParameters
from net_utils import individualParamsSave

import eddl_compss_distributed_api as compss_api

from cvars import *

def main(args):
    print("E: ", platform.uname())

    eddl.download_mnist()

    start_time = timer()

    num_workers = args.num_workers
    num_epochs = args.num_epochs
    workers_batch_size = args.workers_batch_size
    num_epochs_for_param_sync = args.num_epochs_for_param_sync
    max_num_async_epochs = args.max_num_async_epochs

    num_classes = 10

    # Model that works
    in_ = eddl.Input([784])

    layer = in_
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    print(eddl.summary(net))

    compss_api.build(
        net,
        eddl.sgd(CVAR_SGD1, CVAR_SGD2),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(),
        True
    )

    x_train = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_X_TRN)
    y_train = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_Y_TRN)
    x_test = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_X_TST)
    y_test = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_Y_TST)

    eddlT.div_(x_train, 255.0)
    eddlT.div_(x_test, 255.0)

    # Model training
    print("Model training...")
    print("Number of epochs: ", num_epochs)
    print("Number of epochs for parameter syncronization: ", num_epochs_for_param_sync)
    
    for _ in range(0, 1):
        for i in range(0, num_epochs):

            #start_epoch = num_epochs_for_param_sync * i + 1
            #end_epoch = start_epoch + num_epochs_for_param_sync - 1

            #print("Training epochs [", start_epoch, " - ", end_epoch, "] ...")
            print("Training epoch: ", i)
            compss_api.train_batch(net, x_train, y_train, num_workers, num_epochs_for_param_sync, workers_batch_size)

            # Model evaluation
            p = net_parametersToNumpy(net.getParameters())
            print("Freq: ", getFreqFromParameters(p))

            individualParamsSave(p, "sync" + str(num_epochs) + "-epoch" + str(i) + ".txt")

            print("Evaluating model against train set")
            eddl.evaluate(net, [x_train], [y_train])

            print("Evaluating model against test set")
            eddl.evaluate(net, [x_test], [y_test])


    end_time = timer()
    final_time = end_time - start_time

    print("Elapsed time: ", final_time, " seconds")

    # Model evaluation
    '''p = net_parametersToNumpy(net.getParameters())
    print("Freq: ", getFreqFromParameters(p))

    individualParamsSave(p, "sync" + str(num_epochs) + ".txt")
    print("Evaluating model against test set")
    eddl.evaluate(net, [x_test], [y_test])'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_workers", type=int, metavar="INT", default=4)
    parser.add_argument("--num_epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--num_epochs_for_param_sync", type=int, metavar="INT", default=1)
    parser.add_argument("--max_num_async_epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--workers_batch_size", type=int, metavar="INT", default=250)
    #parser.add_argument("--epochs", type=int, metavar="INT", default=4)
    #parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    #parser.add_argument("--gpu", action="store_true")

    main(parser.parse_args(sys.argv[1:]))
