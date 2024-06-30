import pandas as pd
pd.set_option('display.max_rows', None)
df = pd.read_csv(r'C:\Users\edith\Thesis_nieuw\Python workfiles\building\wordprod_full')

#print(len(df[df['correlation'] >0.2]['correlation']))
df['corr_thres']=0

df['corr_thres'] = df['correlation'].apply(lambda x: 1 if x > 0.2 else 0)

print(df)
df.to_csv('wordprod_corr_thres.csv', index=False)