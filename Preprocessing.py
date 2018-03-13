import numpy as np
import pandas as pd

# Read dataset file ckd.csv
dataset = pd.read_csv("ckd.csv",header=0, na_values="?")

# Replace null values "?" by numpy.NaN
dataset.replace("?", np.NaN)

# Convert nominal values to binary values
cleanup = {"Rbc":     {"normal": 1, "abnormal": 0},
           "Pc": {"normal": 1, "abnormal": 0},
           "Pcc": {"present": 1, "notpresent": 0},
           "Ba": {"present": 1, "notpresent": 0},
           "Htn": {"yes": 1, "no": 0},
           "Dm": {"yes": 1, "no": 0},
           "Cad": {"yes": 1, "no": 0},
           "Appet": {"good": 1, "poor": 0},
           "pe": {"yes": 1, "no": 0},
           "Ane": {"yes": 1, "no": 0}}

# Replace binary values into dataset
dataset.replace(cleanup, inplace=True)

# Fill null values with mean value of the respective column

dataset.fillna(round(dataset.mean(),2), inplace=True)

# print(dataset)

# Save this dataset as final.csv for further prediction
dataset.to_csv("final.csv", sep=',', index=False)
