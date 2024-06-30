from os import remove
import numpy as np
import subprocess
import pandas as pd
import glob
pd.set_option('display.max_rows', None)

spec = np.load(r'C:\Users\edith\Thesis\SingleWordProductionDutch\features\sub-02_spec.npy')
envelope = np.mean(spec,axis=1)
feat = np.load(r'C:\Users\edith\Thesis\SingleWordProductionDutch\features\sub-02_feat.npy')


allCorrelations = np.zeros(feat.shape[1])
for i in range(feat.shape[1]):
    r = np.corrcoef(envelope, feat[:, i])
    allCorrelations[i] = r[0, 1]

#print(allCorrelations)

def fetch_x_coordinate(subject_id, feat, feat_names):
    # Load x-coordinates from a corresponding .npy file
    x_coords = np.load(r'C:\Users\edith\Thesis\Python workfiles\x_array.npy', allow_pickle=True)
    # Assume feat_names and x_coords are aligned and have the same order
    index = np.where(feat_names == feat)[0][0]  # Get the index of the feat_name
    return x_coords[index]

corr = []

# Load participant data outside the loop for efficiency
df2 = pd.read_csv(r'C:\Users\edith\Thesis\Python workfiles\wordprod-full.csv')
df2['correlation']=0.0

for i in range(1, 11):
    spec = np.load(r'SingleWordProductionDutch/features/sub-%02d_spec.npy' % i, allow_pickle=True)
    feat_names = np.load(r'C:\Users\edith\Thesis\SingleWordProductionDutch\features\sub-%02d_feat_names.npy' % i, allow_pickle=True)
    feat = np.load(r'SingleWordProductionDutch/features/sub-%02d_feat.npy' % i, allow_pickle=True)
    
    envelope = np.mean(spec, axis=1)
    
    for j in range(feat.shape[1]):
        x_coord = fetch_x_coordinate(i, feat_names[j], feat_names)
        subject_name = ("sub-%02d" % i)
        electrode_name = feat_names[j]

        r = np.corrcoef(envelope, feat[:, j])[0, 1]  # Extract the correlation coefficient
        df2.loc[(df2['name'] == feat_names[j]) & (df2['participant_id'] == subject_name),"correlation"] = r
        #df2.loc[row,col]["correlation"] = r
        
print(df2.head(200))

# Convert to DataFrame and save
pd.set_option('display.max_rows', None)
df2.to_csv('wordprod_corr.csv', index=False)
#print(df.head(200))

