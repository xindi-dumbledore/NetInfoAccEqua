import numpy as np
M, E, N = 0.2, 2, 5000
H, ALPHA = 0.8, 1
PD, ED = 0.6, 1

N_TRIALS = 10
N_GRAPHS = 20
BETA_ARRAY_SYM = np.array([[0.7, 0.7], [0.7, 0.7]])
BETA_ARRAY_ASY = np.array([[0.7, 0.3], [0.3, 0.7]])
GAMMA = 0.1
SEED_NUM = 10

MINORITY_SEEDING_PORTION_DICT = {
    "low": [0, 0.3], "mid": [0.3, 0.7], "high": [0.7, 1]}
