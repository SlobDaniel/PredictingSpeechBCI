import pandas as pd
import tabulate as tabulate
import glob

df1 = pd.read_csv(
    r'/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS/sub-01/ieeg/sub-01_task-wordProduction_channels.tsv')
print("Pandas version:", pd.__version__)

#df2.to_csv('sub-01-task-channels-conv.csv', index=False)
#df2.to_csv('sub-02-task-channels-conv.csv', index=False)
#df2.to_csv('sub-03-task-channels-conv.csv', index=False)
#df2.to_csv('sub-04-task-channels-conv.csv', index=False)
#df2.to_csv('sub-05-task-channels-conv.csv', index=False)
#df2.to_csv('sub-06-task-channels-conv.csv', index=False)
#df2.to_csv('sub-07-task-channels-conv.csv', index=False)
#df2.to_csv('sub-08-task-channels-conv.csv', index=False)
#df2.to_csv('sub-09-task-channels-conv.csv', index=False)
#df2.to_csv('sub-10-task-channels-conv.csv', index=False)

path = r'/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS/conversion_csv'
tsvfiles = glob.glob(path + "/*.tsv")
for t in tsvfiles:
    tsv = pd.read_table(t, sep='\t')
    tsv.to_csv(t[:-4] + '.csv', index=False)

