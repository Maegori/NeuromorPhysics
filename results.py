import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
# %matplotlib inline

import keras
from keras.models import Model
from keras.layers import Dropout, Flatten, Conv2D, Input
from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler 

sys.path.append("/homes/lexjohan/Documents/EME/SVDD/")
from dataloader import unpack, unpack_ordered, unpack_polina
from modeldefault import VariationalAutoencoderModel

sys.path.append('/homes/lexjohan/Documents/models')
os.path.join("nxsdk_modules_ncl/snntoolbox")
from nxsdk_modules_ncl.dnn.src.utils import extract, to_integer
from nxsdk_modules_ncl.dnn.src.dnn_layers import NxInputLayer, NxDense, \
    NxModel, ProbableStates
from nxsdk_modules_ncl.dnn.composable.composable_dnn import ComposableDNN as DNN
from nxsdk_modules_ncl.input_generator.input_generator import InputGenerator
import nxsdk
from nxsdk.composable.model import Model
from nxsdk.logutils.nxlogging import set_verbosity,LoggingLevel
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition

# #environment variables
os.environ['SLURM'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

os.environ["PARTITION"] = "nahuku32_2h"
os.environ["BOARD"] = 'ncl-ext-ghrd-01'

#parser
parser = ArgumentParser()
parser.add_argument('--zdim', type=str, default='5', help='5, 8, 13, 21, 24, 55, 89, 144')
parser.add_argument('--data_split', type=float, default=0.1)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--steps_per_sample', type=int, default=25)
args = parser.parse_args()

def run_model(num_steps_per_sample, train_data, dim_z):
    batch_size = 64
    num_samples = train_data.shape[0]

    tStart = time.time()
    snn_model.run(num_steps_per_sample * num_samples, aSync=True)
    tEndBoot = time.time()

    with tqdm(total=num_samples) as pbar:
        for idx in range(0, num_samples, batch_size):
            batch = train_data[idx:idx+batch_size]
            b = np.abs(np.rint(batch).astype(int))
            input_generator.batchEncode(b)
            pbar.update(batch_size)

    tEndInput = time.time()

    out = list(dnn.readout_channel.read(num_samples, dim_z))
    tEndClassification = time.time()

    snn_model.finishRun()
    
    return tStart, tEndBoot, tEndInput, tEndClassification, out

#load data
mode = 'ordered' #mode = 'polina' / mode = 'None'
Flags = None

data_path = "../EME/SVDD/data/"

train_path = data_path + "training.h5"
test_path = data_path + "testing.h5"

if mode == 'ordered':
    _, regression = unpack_ordered(train_path, Flags)
elif mode == 'polina':
    _, num_objects, regression = unpack_polina(train_path, Flags)
else:
    _, num_objects, regression, classification = unpack(train_path, Flags)

scaler = StandardScaler()

scaler.fit(regression)
reg_normalized = scaler.transform(regression)

if mode == 'ordered':
    training_data = reg_normalized[:]
elif mode == 'polina':
    training_data = [num_objects[:], reg_normalized[:]]
else:
    training_data = [classification[:], reg_normalized[:]]

dataset_len = regression.shape[0]
if mode == 'ordered' or mode == 'polina':
    data_dim = regression.shape[1]
else:
    data_dim = classification.shape[1]


#setup model
models_path = "../EME/SVDD/optimized_W/"
model_name = "SVDD_3l_512_256_128_bs_10000_ordered_ft_0_zdim_" + str(args.zdim) + ".h5"
dim_z = int(args.zdim)
ft = 0

hidden_layers = [512, 256, 128]

ann_model = VariationalAutoencoderModel(hidden_layers, model_name, data_dim, dataset_len, dim_z, ft, mode=mode, verbose=True)
ann_model.load_weights(models_path + model_name)

#create snn model

num_steps = 2

vth_mant = 2**9
bias_exp = 6
weight_exponent = 0
synapse_encoding = 'sparse'

in_regression = NxInputLayer(data_dim,
                            vThMant=vth_mant,
                            biasExp=bias_exp)

layer = NxDense(hidden_layers[0])(in_regression.input)
for idx in range(1, len(hidden_layers)):
    layer = NxDense(hidden_layers[idx])(layer)

layer = NxDense(dim_z)(layer)

set_verbosity(LoggingLevel.ERROR)
# Extract weights and biases from parameter list.
parameters = ann_model.model.get_weights()
weights = parameters[0::2]
biases = parameters[1::2]

# Quantize weights and biases using max-normalization (Strong quantization loss if distributions have large tails)
parameters_int = []
for w, b in zip(weights, biases):
    w_int, b_int = to_integer(w, b, 8)
    parameters_int += [w_int, b_int]

num_steps_per_img = args.steps_per_sample
data_dim_snn = (data_dim, )

data_split = 0.01
num_test_samples = int(training_data.shape[0] * data_split)
execution_time_probe_bin_size = 512 # Small numbers slows down execution

avg_time_per_sample = []
avg_energy_per_sample = []
avg_total_energy = []
avg_total_time = []
output = []

num_steps = args.num_steps
for n in range(num_steps):
    print("Step:", n)

    snn_nxmodel = NxModel(in_regression.input, layer)
    # Set quantized weigths and biases for spiking model
    snn_nxmodel.set_weights(parameters_int)

    dnn = DNN(model=snn_nxmodel, num_steps_per_img=num_steps_per_img)

    input_generator = InputGenerator(shape=data_dim_snn, interval=num_steps_per_img)
    input_generator.setBiasExp(bias_exp)

    snn_model = nxsdk.composable.model.Model("dnn_model")
    snn_model.add(dnn)
    snn_model.add(input_generator)

    input_generator.connect(dnn)

    input_generator.processes.inputEncoder.executeAfter(dnn.processes.reset)
    snn_model.compile()

    eProbe = snn_model.board.probe(
        probeType=ProbeParameter.ENERGY, 
        probeCondition=PerformanceProbeCondition(
            tStart=1, 
            tEnd=num_test_samples*num_steps_per_img, 
            bufferSize=1024, 
            binSize=execution_time_probe_bin_size))

    snn_model.start(snn_nxmodel.board)
    tStart, tEndBoot, tEndInput, tEndClassification, out = run_model(num_steps_per_img, training_data[:num_test_samples], dim_z)
    snn_model.disconnect()

    #collect average values
    avg_time_per_sample.append((tEndClassification - tStart) / num_test_samples)
    avg_energy_per_sample.append(eProbe.totalEnergy / num_test_samples)
    avg_total_energy.append(eProbe.totalEnergy)
    avg_total_time.append(tEndClassification - tStart)
    output.append(out)

# tStart, tEndBoot, tEndInput, tEndClassification = run_model(num_steps_per_img, training_data)

#save results
results_path = "results/"
if not os.path:
    os.mkdir(results_path)

with open(results_path+f'output_{dim_z}_{args.steps_per_sample}.pkl', 'wb') as f:
    np.save(f, np.array(output))

with open(results_path+f"stats_{dim_z}_{args.steps_per_sample}.pkl", 'wb') as f:
    pickle.dump({'avg_time_per_sample': avg_time_per_sample, 
                'avg_energy_per_sample': avg_energy_per_sample, 
                'avg_total_energy': avg_total_energy, 
                'avg_total_time': avg_total_time}, f)

# with open(results_path+f'output_{zdim}_{args.steps_per_sample}.pkl', 'wb') as f:
#     np.save(f, np.array(out))

# energy_dict = dict()
# energy_dict['tStart'] = tStart
# energy_dict['tEndBoot'] = tEndBoot
# energy_dict['tEndInput'] = tEndInput
# energy_dict['tEndClassification'] = tEndClassification
# energy_dict['totalEnergy'] = eProbe.totalEnergy
# energy_dict['energyUnits'] = eProbe.energyUnits

#     #save energy
#     with open(results_path+f'energy_{zdim}_{args.steps_per_sample}.pkl', 'wb') as f:
#         pickle.dump(energy_dict, f)
