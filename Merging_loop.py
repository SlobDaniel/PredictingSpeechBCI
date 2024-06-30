import glob
import pandas as pd

#d = glob.glob(
#    r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\W "
#    r"id\sub-*_task-wordProduction_*-id.csv"
#)


df_ = []
for file in glob.glob(
    r"C:\Users\edith\Thesis\SingleWordProductionDutch\SingleWordProductionDutch-iBIDS\conversion_csv\W "
    r"id\sub-*_task-wordProduction_channels-id.csv"
):
    new_df = pd.read_csv(file)
    df_.append(new_df)

df = pd.concat(df_)
print(df)

df.to_csv('wordprod-channels.csv', index=False)
