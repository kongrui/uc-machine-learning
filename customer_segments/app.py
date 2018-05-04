# Import libraries necessary for this project
import numpy as np
import pandas as pd
#from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
#import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

data = pd.read_csv("customers.csv")
data.drop(['Region', 'Channel'], axis = 1, inplace = True)
