import sys
import pandas as pd
from scipy.fftpack import dct
import numpy as np
sys.path.extend('/Users/sifanliu/Documents/GitHub/Experiments')
from Sketching_methods import hadamard_projection


n = 1000
p = 100
hadamard_projection(100)