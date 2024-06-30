# Merge dataframes on 'patient_id'
import glob
import pandas as pd

# d = glob.glob(r'C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-*_task-wordProduction_channels.csv')

##patient characteristics are targeted
pt_df = pd.read_csv(
    r'C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\participants.csv')

##electrode location df's are created and gathered in a dictionary
#electrode_dict = {
   # 1: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-01_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #2: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-02_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #3: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-03_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #4: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-04_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #5: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-05_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #6: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-06_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #7: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-07_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #8: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-08_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #9: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-09_task-wordProduction_space-ACPC_electrodes.csv")
    #,
    #10: pd.read_csv(
    #r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-10_task-wordProduction_space-ACPC_electrodes.csv")
#}

#for participant_id, df1 in electrode_dict.items():
    #df1['participant_id'] = participant_id

##Ònly df10 is downloaded

df_wp1 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-01_task-wordProduction_channels.csv")
df_wp2 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-02_task-wordProduction_channels.csv")
df_wp3 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-03_task-wordProduction_channels.csv")
df_wp4 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-04_task-wordProduction_channels.csv")
df_wp5 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-05_task-wordProduction_channels.csv")
df_wp6 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-06_task-wordProduction_channels.csv")
df_wp7 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-07_task-wordProduction_channels.csv")
df_wp8 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-08_task-wordProduction_channels.csv")
df_wp9 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-09_task-wordProduction_channels.csv")
df_wp10 = pd.read_csv(r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\sub-10_task-wordProduction_channels.csv")

df_wp1['participant_id'] = 'sub-01'
df_wp1.to_csv('sub-01_task-wordProduction_channels-id.csv', index=False)

df_wp2['participant_id'] = 'sub-02'
df_wp2.to_csv('sub-02_task-wordProduction_channels-id.csv', index=False)

df_wp3['participant_id'] = 'sub-03'
df_wp3.to_csv('sub-03_task-wordProduction_channels-id.csv', index=False)

df_wp4['participant_id'] = 'sub-04'
df_wp4.to_csv('sub-04_task-wordProduction_channels-id.csv', index=False)

df_wp5['participant_id'] = 'sub-05'
df_wp5.to_csv('sub-05_task-wordProduction_channels-id.csv', index=False)

df_wp6['participant_id'] = 'sub-06'
df_wp6.to_csv('sub-06_task-wordProduction_channels-id.csv', index=False)

df_wp7['participant_id'] = 'sub-07'
df_wp7.to_csv('sub-07_task-wordProduction_channels-id.csv', index=False)

df_wp8['participant_id'] = 'sub-08'
df_wp8.to_csv('sub-08_task-wordProduction_channels-id.csv', index=False)

df_wp9['participant_id'] = 'sub-09'
df_wp9.to_csv('sub-09_task-wordProduction_channels-id.csv', index=False)

df_wp10['participant_id'] = 'sub-10'
df_wp10.to_csv('sub-10_task-wordProduction_channels-id.csv', index=False)

