import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)

df = pd.read_csv(r'C:\Users\edith\Thesis\Python workfiles\EDA\wordprod_corr.csv')

##  Dropping irrelevant columns and rows, see notes for explanation

df2 = df[df['correlation'] != 0][df['description'] != 'Unknown']
print(df2.shape)


##  exploration

#print(len(df2['description'].unique()))
#df_ohe = pd.get_dummies(df2, drop_first=True)


##  building one hot encoding 

print(df2['description'].value_counts().sort_values(ascending=False).head(10))
#top_10 = df2['description'].value_counts().sort_values(ascending=False).head(12).index
top_10 = df2['description'].value_counts().drop(['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter']).head(10).index


for label in top_10:
    df2[label] = np.where(df2['description'] == label, 1, 0)

df_one_hot = df2[['description'] + list(top_10)]

merged_df = df2.merge(df_one_hot, left_index=True, right_index=True, suffixes=('_', '_delete'))

df3 = merged_df.drop(['description_delete', 'ctx_rh_S_temporal_sup_delete', 'ctx_rh_G_pariet_inf-Supramar_delete', 'Right-Hippocampus_delete', 'ctx_rh_Lat_Fis-post_delete', 
                'WM-hypointensities_delete', 'ctx_rh_S_circular_insula_inf_delete', 'Left-Hippocampus_delete', 'ctx_rh_G_temporal_middle_delete', 
                'ctx_lh_G_pariet_inf-Supramar_delete', 'ctx_rh_G_front_inf-Opercular_delete'], axis=1)

print(df3)
df3.to_csv('wordprod_full', )
