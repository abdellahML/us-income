import pandas as pd
import numpy as np

# imports sets
us_income_train = "datasets/cleaned/data_train.csv"
us_income_test = "datasets/cleaned/data_test.csv"
df = pd.read_csv(us_income_train)
df_test = pd.read_csv(us_income_test)
