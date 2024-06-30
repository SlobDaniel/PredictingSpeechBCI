import pandas as pd

df1 = pd.read_csv(r'C:\Users\edith\Thesis\Python workfiles\wordprod-channels.csv')
df2 = pd.read_csv(r'C:\Users\edith\Thesis\Python workfiles\wordprod-space.csv')

df3 = df1.merge(df2, how='outer')

print(df3)
df3.to_csv('wordprod_space_channels.csv', index=False)