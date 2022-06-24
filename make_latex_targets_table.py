from astropy.table import Table
from astropy.io import ascii
import pandas as pd
import numpy as np

df = pd.read_csv('NHI_data.csv')
df=df.sort_values(by='distance (pc)', ascending=True)
df.reset_index(inplace=True)

for i in range(len(df)):

    df['distance (pc)'].loc[i] = np.round(df['distance (pc)'].loc[i], 2)
    df['RA'].loc[i] = np.round(df['RA'].loc[i], 2)
    df['DEC'].loc[i] = np.round(df['DEC'].loc[i], 2)
    df['N(HI)'].loc[i] = np.round(df['N(HI)'].loc[i], 2)
    df['N(HI) uncertainty'].loc[i] = np.round(df['N(HI) uncertainty'].loc[i], 2)

## find unique references for easy numbering!
reference_list = df['N(HI) source'].unique()
df['N(HI) source numbers'] = 


t2 = Table.from_pandas(df)
t3 = t2['Star Name', 'RA', 'DEC', 'distance (pc)', 'N(HI)', 'N(HI) uncertainty', 'N(HI) source numbers']
ascii.write(t3,format='latex')
