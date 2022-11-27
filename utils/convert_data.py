import numpy as np

def normalize(y):
    return np.div(y, np.norm(y, p=2))

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

n = 16
states = neutral_states(n)
np.savez_compressed("../models/data/x_test.npz", states)

with open("../models/data/ED_amplitude.csv", "r") as file:
    true_wf = np.loadtxt(file, delimiter=",", dtype=np.float64)

np.savez_compressed("../models/data/y_test.npz", true_wf)

