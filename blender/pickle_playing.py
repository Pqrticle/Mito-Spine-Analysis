import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Open the pickle file in 'rb' (read binary) mode
with open('../data/calcium_trace.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)