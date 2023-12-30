import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm, gamma, binom

def chance_level(n, alpha = 0.001, p = 0.5):
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level

def sample_truncated_normal(mean, sd, lower, upper, size=1000):
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

def savage_dickey_plot(data, parameters=[("alpha_a_rew", "$\alpha A_{rew}$"), ("alpha_a_pun", "$\alpha A_{pun}$"), ("alpha_K", "$\alpha K$"), ("alpha_theta","$\alpha \\theta$"), ("alpha_omega_f","$\alpha \omega_F$"), ("alpha_omega_p", "$\alpha \omega_P$")]):
    pass