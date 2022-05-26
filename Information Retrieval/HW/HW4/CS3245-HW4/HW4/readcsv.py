import pandas as pd
import time
start = time.time()
#read data in chunks of 1 million rows at a time
chunk = pd.read_csv('dataset/dataset.csv',chunksize=1000000)
end = time.time()
print("Read csv with chunks: ",(end-start),"sec")
pd_df = pd.concat(chunk)
print('a')
print(pd_df)