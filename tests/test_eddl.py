# Copyright (c) 2019-2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import pyeddl._core.eddl as eddl_core
import pyeddl.eddl as eddl_py


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_core_layers(eddl):
    in2d = eddl.Input([16])
    in4d = eddl.Input([3, 16, 16])
    eddl.Activation(in2d, "relu")
    eddl.Activation(in2d, "relu", [0.02])
    eddl.Activation(in2d, "relu", [0.02], "foo")
    eddl.Softmax(in2d)
    eddl.Softmax(in2d, "foo")
    eddl.Sigmoid(in2d)
    eddl.Sigmoid(in2d, "foo")
    eddl.HardSigmoid(in2d)
    eddl.HardSigmoid(in2d, "foo")
    eddl.ReLu(in2d)
    eddl.ReLu(in2d, "foo")
    eddl.ThresholdedReLu(in2d)
    eddl.ThresholdedReLu(in2d, 1.0)
    eddl.ThresholdedReLu(in2d, 1.0, "foo")
    eddl.LeakyReLu(in2d)
    eddl.LeakyReLu(in2d, 0.02)
    eddl.LeakyReLu(in2d, 0.02, "foo")
    eddl.Elu(in2d)
    eddl.Elu(in2d, 1.0)
    eddl.Elu(in2d, 1.0, "foo")
    eddl.Selu(in2d)
    eddl.Selu(in2d, "foo")
    eddl.Exponential(in2d)
    eddl.Exponential(in2d, "foo")
    eddl.Softplus(in2d)
    eddl.Softplus(in2d, "foo")
    eddl.Softsign(in2d)
    eddl.Softsign(in2d, "foo")
    eddl.Linear(in2d)
    eddl.Linear(in2d, 1.0)
    eddl.Linear(in2d, 1.0, "foo")
    eddl.Tanh(in2d)
    eddl.Tanh(in2d, "foo")
    eddl.Conv(in4d, 16, [1, 1])
    eddl.Conv(in4d, 16, [1, 1], [2, 2], "none")
    eddl.Conv(in4d, 16, [1, 1], [2, 2], "none", True)
    eddl.Conv(in4d, 16, [1, 1], [2, 2], "none", True, 1)
    eddl.Conv(in4d, 16, [1, 1], [2, 2], "none", True, 1, [1, 1])
    eddl.Conv(in4d, 16, [1, 1], [2, 2], "none", True, 1, [1, 1], "foo")
    eddl.ConvT(in4d, 16, [1, 1], [2, 2])
    eddl.ConvT(in4d, 16, [1, 1], [2, 2], "none")
    eddl.ConvT(in4d, 16, [1, 1], [2, 2], "none", [1, 1])
    eddl.ConvT(in4d, 16, [1, 1], [2, 2], "none", [1, 1], [1, 1])
    eddl.ConvT(in4d, 16, [1, 1], [2, 2], "none", [1, 1], [1, 1], True)
    eddl.ConvT(in4d, 16, [1, 1], [2, 2], "none", [1, 1], [1, 1], True, "foo")
    eddl.Dense(in2d, 16)
    eddl.Dense(in2d, 16, True)
    eddl.Dense(in2d, 16, True, "foo")
    eddl.Embedding(in2d, 1001, 16, 250)
    eddl.Embedding(in2d, 1001, 16, 250, True)
    eddl.Embedding(in2d, 1001, 16, 250, True, "foo")
    eddl.Input([16], "foo")
    eddl.UpSampling(in4d, [2, 2])
    eddl.UpSampling(in4d, [2, 2], "nearest")
    eddl.UpSampling(in4d, [2, 2], "nearest", "foo")
    eddl.Reshape(in2d, [1, 4, 4])
    eddl.Reshape(in2d, [1, 4, 4], "foo")
    eddl.Flatten(in2d)
    eddl.Flatten(in2d, "foo")
    eddl.Transpose(in2d)
    eddl.Transpose(in2d, "foo")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_transformations(eddl):
    in2d = eddl.Input([16])
    eddl.Shift(in2d, [1, 1])
    eddl.Shift(in2d, [1, 1], "nearest")
    eddl.Shift(in2d, [1, 1], "nearest", 0.0)
    eddl.Shift(in2d, [1, 1], "nearest", 0.0, "foo")
    eddl.Rotate(in2d, 1.1)
    eddl.Rotate(in2d, 1.1, [0, 0])
    eddl.Rotate(in2d, 1.1, [0, 0], "nearest")
    eddl.Rotate(in2d, 1.1, [0, 0], "nearest", 0.0)
    eddl.Rotate(in2d, 1.1, [0, 0], "nearest", 0.0, "foo")
    eddl.Scale(in2d, [1, 4], True)
    eddl.Scale(in2d, [1, 4], True, "nearest")
    eddl.Scale(in2d, [1, 4], True, "nearest", 0.0)
    eddl.Scale(in2d, [1, 4], True, "nearest", 0.0, "foo")
    eddl.Flip(in2d)
    eddl.Flip(in2d, 0)
    eddl.Flip(in2d, 0, "foo")
    eddl.HorizontalFlip(in2d)
    eddl.HorizontalFlip(in2d, "foo")
    eddl.VerticalFlip(in2d)
    eddl.VerticalFlip(in2d, "foo")
    eddl.Crop(in2d, [0, 0], [3, 3])
    eddl.Crop(in2d, [0, 0], [3, 3], True)
    eddl.Crop(in2d, [0, 0], [3, 3], True, 0.0)
    eddl.Crop(in2d, [0, 0], [3, 3], True, 0.0, "foo")
    eddl.CenteredCrop(in2d, [3, 3])
    eddl.CenteredCrop(in2d, [3, 3], True)
    eddl.CenteredCrop(in2d, [3, 3], True, 0.0)
    eddl.CenteredCrop(in2d, [3, 3], True, 0.0, "foo")
    eddl.CropScale(in2d, [0, 0], [3, 3])
    eddl.CropScale(in2d, [0, 0], [3, 3], "nearest")
    eddl.CropScale(in2d, [0, 0], [3, 3], "nearest", 0.0)
    eddl.CropScale(in2d, [0, 0], [3, 3], "nearest", 0.0, "foo")
    eddl.Cutout(in2d, [0, 0], [3, 3])
    eddl.Cutout(in2d, [0, 0], [3, 3], 0.0)
    eddl.Cutout(in2d, [0, 0], [3, 3], 0.0, "foo")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_data_augmentation(eddl):
    in2d = eddl.Input([16])
    eddl.RandomShift(in2d, [1.0, 1.0], [2.0, 2.0])
    eddl.RandomShift(in2d, [1.0, 1.0], [2.0, 2.0], "nearest")
    eddl.RandomShift(in2d, [1.0, 1.0], [2.0, 2.0], "nearest", 0.0)
    eddl.RandomShift(in2d, [1.0, 1.0], [2.0, 2.0], "nearest", 0.0, "foo")
    eddl.RandomRotation(in2d, [1.0, 1.0])
    eddl.RandomRotation(in2d, [1.0, 1.0], [0, 0])
    eddl.RandomRotation(in2d, [1.0, 1.0], [0, 0], "nearest")
    eddl.RandomRotation(in2d, [1.0, 1.0], [0, 0], "nearest", 0.0)
    eddl.RandomRotation(in2d, [1.0, 1.0], [0, 0], "nearest", 0.0, "foo")
    eddl.RandomScale(in2d, [1.0, 1.0])
    eddl.RandomScale(in2d, [1.0, 1.0], "nearest")
    eddl.RandomScale(in2d, [1.0, 1.0], "nearest", 0.0)
    eddl.RandomScale(in2d, [1.0, 1.0], "nearest", 0.0, "foo")
    eddl.RandomFlip(in2d, 0)
    eddl.RandomFlip(in2d, 0, "foo")
    eddl.RandomHorizontalFlip(in2d)
    eddl.RandomHorizontalFlip(in2d, "foo")
    eddl.RandomVerticalFlip(in2d)
    eddl.RandomVerticalFlip(in2d, "foo")
    eddl.RandomCrop(in2d, [4, 4])
    eddl.RandomCrop(in2d, [4, 4], "foo")
    eddl.RandomCropScale(in2d, [1.0, 1.0])
    eddl.RandomCropScale(in2d, [1.0, 1.0], "nearest")
    eddl.RandomCropScale(in2d, [1.0, 1.0], "nearest", "foo")
    eddl.RandomCutout(in2d, [1.0, 1.0], [1.0, 1.0])
    eddl.RandomCutout(in2d, [1.0, 1.0], [1.0, 1.0], 0.0)
    eddl.RandomCutout(in2d, [1.0, 1.0], [1.0, 1.0], 0.0, "foo")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_losses(eddl):
    eddl.getLoss("mse")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_metrics(eddl):
    eddl.getMetric("mse")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_merge_layers(eddl):
    in_1 = eddl.Input([16])
    in_2 = eddl.Input([16])
    eddl.Add([in_1, in_2])
    eddl.Add([in_1, in_2], "foo")
    eddl.Average([in_1, in_2])
    eddl.Average([in_1, in_2], "foo")
    eddl.Concat([in_1, in_2])
    eddl.Concat([in_1, in_2], 1)
    eddl.Concat([in_1, in_2], 1, "foo")
    eddl.MatMul([in_1, in_2])
    eddl.MatMul([in_1, in_2], "foo")
    eddl.Maximum([in_1, in_2])
    eddl.Maximum([in_1, in_2], "foo")
    eddl.Minimum([in_1, in_2])
    eddl.Minimum([in_1, in_2], "foo")
    eddl.Subtract([in_1, in_2])
    eddl.Subtract([in_1, in_2], "foo")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_noise_layers(eddl):
    in2d = eddl.Input([16])
    eddl.GaussianNoise(in2d, 1.1)
    eddl.GaussianNoise(in2d, 1.1, "foo")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_normalization_layers(eddl):
    in2d = eddl.Input([16])
    eddl.BatchNormalization(in2d, True)
    eddl.BatchNormalization(in2d, True, 0.9)
    eddl.BatchNormalization(in2d, True, 0.9, 0.001)
    eddl.BatchNormalization(in2d, True, 0.9, 0.001)
    eddl.BatchNormalization(in2d, True, 0.9, 0.001, "foo")
    eddl.LayerNormalization(in2d, True)
    eddl.LayerNormalization(in2d, True, 0.001)
    eddl.LayerNormalization(in2d, True, 0.001, "foo")
    # Norm{,Max,MinMax} tests now lead to a "RuntimeError: axis 1 >= dim=1"
    # eddl.Norm(in2d)
    # eddl.Norm(in2d, 0.001)
    # eddl.Norm(in2d, 0.001, "foo")
    # eddl.NormMax(in2d)
    # eddl.NormMax(in2d, 0.001)
    # eddl.NormMax(in2d, 0.001, "foo")
    # eddl.NormMinMax(in2d)
    # eddl.NormMinMax(in2d, 0.001)
    # eddl.NormMinMax(in2d, 0.001, "foo")
    eddl.Dropout(in2d, 0.5)
    eddl.Dropout(in2d, 0.5, False, "foo")
    eddl.Dropout(in2d, 0.5, False, "foo")
    in4d = eddl.Input([6, 16, 16])
    eddl.GroupNormalization(in4d, 3)
    eddl.GroupNormalization(in4d, 3, 0.001)
    eddl.GroupNormalization(in4d, 3, 0.001, True)
    eddl.GroupNormalization(in4d, 3, 0.001, True, "foo")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_operator_layers(eddl):
    in_1 = eddl.Input([16])
    in_2 = eddl.Input([16])
    eddl.Abs(in_1)
    eddl.Diff(in_1, in_2)
    eddl.Diff(in_1, 1.0)
    eddl.Diff(1.0, in_1)
    eddl.Div(in_1, in_2)
    eddl.Div(in_1, 1.0)
    eddl.Div(1.0, in_1)
    eddl.Exp(in_1)
    eddl.Log(in_1)
    eddl.Log2(in_1)
    eddl.Log10(in_1)
    eddl.Mult(in_1, in_2)
    eddl.Mult(in_1, 1.0)
    eddl.Mult(1.0, in_1)
    eddl.Pow(in_1, in_2)
    eddl.Pow(in_1, 1.0)
    eddl.Sqrt(in_1)
    eddl.Sum(in_1, in_2)
    eddl.Sum(in_1, 1.0)
    eddl.Sum(1.0, in_1)
    in4d = eddl.Input([3, 16, 16])
    eddl.Select(in4d, [":", ":8", ":8"])
    eddl.Permute(in4d, [0, 1, 2])


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_reduction_layers(eddl):
    in2d = eddl.Input([16])
    eddl.ReduceMean(in2d, [0])
    eddl.ReduceMean(in2d, [0], False)
    eddl.ReduceVar(in2d, [0])
    eddl.ReduceVar(in2d, [0], False)
    eddl.ReduceSum(in2d, [0])
    eddl.ReduceSum(in2d, [0], False)
    eddl.ReduceMax(in2d, [0])
    eddl.ReduceMax(in2d, [0], False)
    eddl.ReduceMin(in2d, [0])
    eddl.ReduceMin(in2d, [0], False)


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_generator_layers(eddl):
    eddl.GaussGenerator(0.0, 1.0, [10])
    eddl.UniformGenerator(0.0, 5.0, [10])


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_optimizers(eddl):
    eddl.adadelta(0.01, 0.9, 0.0001, 0.0)
    eddl.adam()
    eddl.adam(0.01)
    eddl.adam(0.01, 0.9)
    eddl.adam(0.01, 0.9, 0.999)
    eddl.adam(0.01, 0.9, 0.999, 0.0001)
    eddl.adam(0.01, 0.9, 0.999, 0.0001, 0.0)
    eddl.adam(0.01, 0.9, 0.999, 0.0001, 0.0, False)
    eddl.adagrad(0.01, 0.0001, 0.0)
    eddl.adamax(0.01, 0.9, 0.999, 0.0001, 0.0)
    eddl.nadam(0.01, 0.9, 0.999, 0.0001, 0.0)
    eddl.rmsprop()
    eddl.rmsprop(0.01)
    eddl.rmsprop(0.01, 0.9)
    eddl.rmsprop(0.01, 0.9, 0.0001)
    eddl.rmsprop(0.01, 0.9, 0.0001, 0.0)
    eddl.sgd()
    eddl.sgd(0.01)
    eddl.sgd(0.01, 0.0)
    eddl.sgd(0.01, 0.0, 0.0)
    eddl.sgd(0.01, 0.0, 0.0, False)


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_pooling_layers(eddl):
    in4d = eddl.Input([3, 16, 16])
    eddl.AveragePool(in4d)
    eddl.AveragePool(in4d, [2, 2])
    eddl.AveragePool(in4d, [2, 2], [2, 2])
    eddl.AveragePool(in4d, [2, 2], [2, 2], "none")
    eddl.AveragePool(in4d, [2, 2], [2, 2], "none", "foo")
    eddl.MaxPool(in4d)
    eddl.MaxPool(in4d, [2, 2])
    eddl.MaxPool(in4d, [2, 2], [2, 2])
    eddl.MaxPool(in4d, [2, 2], [2, 2], "none")
    eddl.MaxPool(in4d, [2, 2], [2, 2], "none", "foo")
    eddl.GlobalMaxPool(in4d)
    eddl.GlobalMaxPool(in4d, "foo")
    eddl.GlobalAveragePool(in4d)
    eddl.GlobalAveragePool(in4d, "foo")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_recurrent_layers(eddl):
    in2d = eddl.Input([16])
    eddl.RNN(in2d, 1)
    eddl.RNN(in2d, 1, "tanh")
    eddl.RNN(in2d, 1, "tanh", True)
    eddl.RNN(in2d, 1, "tanh", True, True)
    eddl.RNN(in2d, 1, "tanh", True, True, "foo")
    eddl.LSTM(in2d, 1)
    eddl.LSTM(in2d, 1, True)
    eddl.LSTM(in2d, 1, True, True)
    eddl.LSTM(in2d, 1, True, True, "foo")


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_initializers(eddl):
    in2d = eddl.Input([16])
    eddl.GlorotNormal(in2d)
    eddl.GlorotNormal(in2d, 1234)
    eddl.GlorotUniform(in2d)
    eddl.GlorotUniform(in2d, 1234)
    eddl.RandomNormal(in2d)
    eddl.RandomNormal(in2d, 0.0)
    eddl.RandomNormal(in2d, 0.0, 0.1)
    eddl.RandomNormal(in2d, 0.0, 0.1, 1234)
    eddl.RandomUniform(in2d)
    eddl.RandomUniform(in2d, 0.0)
    eddl.RandomUniform(in2d, 0.0, 0.1)
    eddl.RandomUniform(in2d, 0.0, 0.1, 1234)
    eddl.Constant(in2d)
    eddl.Constant(in2d, 0.1)


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_regularizers(eddl):
    in2d = eddl.Input([16])
    eddl.L2(in2d, 0.001)
    eddl.L1(in2d, 0.001)
    eddl.L1L2(in2d, 0.001, 0.001)


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_computing_services(eddl):
    eddl.CS_CPU()
    eddl.CS_CPU(1)
    eddl.CS_CPU(1, "low_mem")
    if eddl is eddl_py:
        eddl.CS_GPU()
    eddl.CS_GPU([1])
    eddl.CS_GPU([1], 1)
    eddl.CS_GPU([1], 1, "low_mem")
    eddl.CS_FGPA([1])
    eddl.CS_FGPA([1], 1)
    eddl.CS_COMPSS("foo.xml")
