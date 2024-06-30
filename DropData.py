import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\edith\Thesis_nieuw\Python workfiles\building\wordprod_full')


df = df.replace({'F': 0}, regex=True)
df = df.replace({'M': 1}, regex=True)

print(df.describe())