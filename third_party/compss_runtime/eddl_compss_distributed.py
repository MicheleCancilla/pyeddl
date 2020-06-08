from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import *
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT

import numpy as np
from net_utils import net_parametersToNumpy
from net_utils import net_parametersToTensor

from cvars import *

class Eddl_Compss_Distributed:

	def __init__(self):
		self.model = None

	@task(serialized_model=IN, optimizer=IN, losses=IN, metrics=IN, compserv=IN, is_replicated=True)
	def build(self, serialized_model, optimizer, losses, metrics, compserv):

		# Deserialize the received model
		model = eddl.import_net_from_onnx_string(serialized_model)

		#print(eddl.summary(model))

		# Build the model in this very node
		eddl.build(
			model,
			eddl.sgd(CVAR_SGD1, CVAR_SGD2),
			losses,
			metrics,
			eddl.CS_CPU(mem="full_mem"),
			False
		)

		# Save the model. We have to serialize it to a string so COMPSs is able to serialize and deserialize from disk
		self.model = eddl.serialize_net_to_onnx_string(model, False)


	@task(initial_parameters=IN, num_images_per_worker=IN, num_epochs_for_param_sync=IN, workers_batch_size=IN, target_direction=IN)
	def train_batch(self, initial_parameters, num_images_per_worker, num_epochs_for_param_sync, workers_batch_size):

		# Deserialize from disk
		model = eddl.import_net_from_onnx_string(self.model)

		# Set the parameters sent from master to the model
		model.setParameters(net_parametersToTensor(initial_parameters))

		#print(eddl.summary(model))

		# The model needs to be built after deserializing
		eddl.build(
			model,
			eddl.sgd(CVAR_SGD1, CVAR_SGD2),
			["soft_cross_entropy"],
			["categorical_accuracy"],
			eddl.CS_CPU(mem="full_mem"),
			False
		)

		print("Build completed in train batch task")

		x_train = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_X_TRN)
		y_train = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_Y_TRN)
		x_test = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_X_TST)
		y_test = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_Y_TST)

		eddlT.div_(x_train, 255.0)
		eddlT.div_(x_test, 255.0)

		s = eddlT.getShape(x_train)
		print("S[0]", s[0])
		num_batches = int(num_images_per_worker / workers_batch_size)

		eddl.reset_loss(model)
		print("Num batches: ", num_batches)

		for i in range(num_epochs_for_param_sync):
			
			print("Epoch %d/%d (%d batches)" % (i + 1, num_epochs_for_param_sync, num_batches))

			for j in range(num_batches):
				indices = np.random.randint(0, s[0], workers_batch_size)
				eddl.train_batch(model, [x_train], [y_train], indices)
				eddl.print_loss(model, j)
				print()

		print("Train batch individual completed in train batch task")

		# Get parameters from the model and convert them to numpy so COMPSS can serialize them
		final_parameters = net_parametersToNumpy(model.getParameters())

		return final_parameters


	@task(initial_parameters=IN, num_images_per_worker=IN, num_epochs_for_param_sync=IN, workers_batch_size=IN, target_direction=IN)
	def train_batch_async(self, initial_parameters, num_images_per_worker, num_epochs_for_param_sync, workers_batch_size):

		# Deserialize from disk
		model = eddl.import_net_from_onnx_string(self.model)

		# Set the parameters sent from master to the model
		model.setParameters(net_parametersToTensor(initial_parameters))

		#print(eddl.summary(model))

		# The model needs to be built after deserializing
		eddl.build(
			model,
			eddl.sgd(CVAR_SGD1, CVAR_SGD2),
			["soft_cross_entropy"],
			["categorical_accuracy"],
			eddl.CS_CPU(mem="full_mem"),
			False
		)

		print("Build completed in train batch task")

		x_train = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_X_TRN)
		y_train = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_Y_TRN)
		x_test = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_X_TST)
		y_test = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_Y_TST)

		eddlT.div_(x_train, 255.0)
		eddlT.div_(x_test, 255.0)

		s = eddlT.getShape(x_train)
		print("S[0]", s[0])
		num_batches = int(num_images_per_worker / workers_batch_size)

		eddl.reset_loss(model)
		print("Num batches: ", num_batches)

		for i in range(num_epochs_for_param_sync):
			
			print("Epoch %d/%d (%d batches)" % (i + 1, num_epochs_for_param_sync, num_batches))

			for j in range(num_batches):
				indices = np.random.randint(0, s[0], workers_batch_size)
				eddl.train_batch(model, [x_train], [y_train], indices)
				eddl.print_loss(model, j)
				print()

		print("Train batch individual completed in train batch task")

		# Get parameters from the model and convert them to numpy so COMPSS can serialize them
		final_parameters = net_parametersToNumpy(model.getParameters())

		return final_parameters


	@task(accumulated_parameters=COMMUTATIVE, parameters_to_aggregate=IN, mult_factor=IN, target_direction=IN)
	def aggregate_parameters_async(self, accumulated_parameters, parameters_to_aggregate, mult_factor):

		for i in range(0, len(accumulated_parameters)):
			for j in range(0, len(accumulated_parameters[i])):
				
				#accumulated_parameters[i][j] += (parameters_to_aggregate[i][j] * mult_factor).astype(np.float32)
				accumulated_parameters[i][j] = ((accumulated_parameters[i][j] + parameters_to_aggregate[i][j]) / 2).astype(np.float32)

		return accumulated_parameters