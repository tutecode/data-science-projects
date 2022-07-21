import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import read_csv
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

# https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset/notebook

# Lets load the dataset and sample some
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv('housing.csv',
                header=None,
                delimiter=r"\s+",
                names=column_names)

print(data.head(5))

# Dimension of the dataset
print(np.shape(data))

# Let's summarize the data to see the distribution of data
print(data.describe())

