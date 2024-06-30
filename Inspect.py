import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\edith\Thesis_nieuw\Python workfiles\building\wordprod_corr_thres.csv')
columns_to_check = ['ctx_rh_S_temporal_sup_',
                    'ctx_rh_G_pariet_inf-Supramar_',
                    'Right-Hippocampus_',
                    'ctx_rh_Lat_Fis-post_',
                    'WM-hypointensities_',
                    'ctx_rh_S_circular_insula_inf_',
                    'Left-Hippocampus_',
                    'ctx_rh_G_temporal_middle_',
                    'ctx_lh_G_pariet_inf-Supramar_',
                    'ctx_rh_G_front_inf-Opercular_']

# Print the count of each unique value for each specified column
for column in columns_to_check:
    print(f'Column {column} value counts:')
    print(df[column].value_counts())
    print('\n')