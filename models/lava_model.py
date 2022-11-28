import lava.lib.dl.slayer as slayer
import torch
import numpy as np

n = 16 # number of spins
alpha = 1 # number of hidden units per spin

neuron_params = {
    'threshold': 1.25,
    'current_decay': 0.25,
    'voltage_decay': 0.03,
}


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = slayer.block.cuba.Dense(neuron_params, n, n * alpha)

    def forward(self, spike):
        y = self.l1(spike)
        y = torch.exp(torch.sum(torch.log(2 * torch.cosh(y)), 1))
        return y

def normalize(y):
    return torch.div(y, torch.norm(y, p=2))

def dec_to_bin(x, y):  # converts a digit to a binary with y entrees
    return format(x, "0{0}b".format(y))

def neutral_states(n):  # gives all spin convigurations of n spins with total spin 0
    states = np.zeros([2**n, n], dtype=np.float64)
    for i in range(0, 2**n, 1):
        for j in range(0, n, 1):
            states[i, j] = 2 * int(dec_to_bin(i, n)[j]) - 1
    neutral_states = np.empty([0, n], dtype=np.float64)
    for i in range(0, 2**n, 1):
        if sum(states[i, :]) == 0:
            neutral_states = np.vstack((neutral_states, states[i,:]))
    return neutral_states

rbm = Model()
rbm.load_state_dict(torch.load("pytorch_model_a1.pkl"))

states = torch.tensor(neutral_states(n))

rbm.eval()

with torch.no_grad():
    for i in range(100000):
        out = rbm.forward(states)

print(out)



