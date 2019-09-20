"""load breast cancer"""
import numpy as np

def load_data(file_path, delimiter, skiprows=0):
    """loads a data file and returns a numpy array"""
    file = open(file_path, "rb")
    arr = np.loadtxt(file, delimiter=delimiter, skiprows=skiprows)
    return arr

load_data("breast-cancer-wisconsin.csv", ",")
