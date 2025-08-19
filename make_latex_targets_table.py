from astropy.table import Table
from astropy.io import ascii
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from math import log10, floor

def round_to_1(x, depth=2):
    if x ==0:
        return 0
    else:
        factor = -int(floor(log10(abs(x))))
        if factor<(depth+1):
            factor = depth
        return round(x, factor)

df = pd.read_csv('targets/NHI_data_August2025.csv')
df=df.sort_values(by='distance (pc)', ascending=True)
df.reset_index(inplace=True)

df['nhi'] = 0
df['nhi uncertainty'] = 0
df['nhi uncertainty adjusted'] = 0
df['l'] = 0
df['b'] = 0

for i in range(len(df)):

    df['nhi'].loc[i] = 10**df['N(HI)'].loc[i] / (df['distance (pc)'].loc[i]*3.09e18)
    df['nhi uncertainty'].loc[i] =df['nhi'].loc[i] * np.log(10) * df['N(HI) uncertainty'].loc[i]
    df['nhi uncertainty adjusted'].loc[i] =df['nhi'].loc[i] * np.log(10) * df['N(HI) uncertainty adjusted'].loc[i]

    df['nhi'].loc[i] = round_to_1(df['nhi'].loc[i])
    df['nhi uncertainty'].loc[i] = round_to_1(df['nhi uncertainty'].loc[i])
    df['nhi uncertainty adjusted'].loc[i] = round_to_1(df['nhi uncertainty adjusted'].loc[i])

    c = SkyCoord(df['RA'].loc[i]*u.deg,df['DEC'].loc[i]*u.deg,frame='icrs')

    df['l'].loc[i] = c.galactic.l.value
    df['b'].loc[i] = c.galactic.b.value




    df['distance (pc)'].loc[i] = np.round(df['distance (pc)'].loc[i], 2)
    df['RA'].loc[i] = np.round(df['RA'].loc[i], 2)
    df['DEC'].loc[i] = np.round(df['DEC'].loc[i], 2)
    df['l'].loc[i] = np.round(df['l'].loc[i], 2)
    df['b'].loc[i] = np.round(df['b'].loc[i], 2)

    df['N(HI)'].loc[i] = np.round(df['N(HI)'].loc[i], 2)
    
    df['N(HI) uncertainty'].loc[i] = round_to_1(df['N(HI) uncertainty'].loc[i])
    df['N(HI) uncertainty adjusted'].loc[i] = round_to_1(df['N(HI) uncertainty adjusted'].loc[i])



## find unique references for easy numbering!
reference_list = df['N(HI) source'].unique()
df['N(HI) source numbers'] = 0

data = {'references': reference_list}
df_tmp = pd.DataFrame(data)

# Dictionary to track numbering
reference_numbers = {}
counter = 1  # Start numbering from 1

# Assign numbers
def assign_number(ref):
    global counter
    if ref not in reference_numbers:
        reference_numbers[ref] = counter
        counter += 1
    return reference_numbers[ref]

df_tmp['reference_number'] = df_tmp['references'].apply(assign_number)

reference_mapping = dict(zip(df_tmp['references'], df_tmp['reference_number']))

df['N(HI) source numbers'] = df['N(HI) source'].map(reference_mapping)

for i in range(len(reference_list)):
  # mask =  df['N(HI) source'] == reference_list[i]
  # df['N(HI) source numbers'][mask] = '[' + str(i+1) + ']'
   print('['+str(i+1)+'] \citealt{'+reference_list[i]+'},')


t2 = Table.from_pandas(df)
t3 = t2['Latex Name', 'l', 'b', 'distance (pc)', 'N(HI)', 'N(HI) uncertainty', 'N(HI) uncertainty adjusted', 'nhi','nhi uncertainty','nhi uncertainty adjusted','Instrument/Grating','N(HI) source numbers']
ascii.write(t3,format='latex')


"""
def D_column_calc(logDs,sig_logDs):
    D = np.sum(10**logDs)
    sig_D_squared = np.sum((sig_logDs*10**logDs)**2)
    return D, np.sqrt(sig_D_squared)
def N_column_calc(D,sig_D,d2h=1.5e-5):
    N = D/d2h
    sig_N = np.sqrt((sig_D/1.5e-5)**2)
    return N, sig_N
def logN_column_calc(N,sig_N):
    logN = np.log10(N)
    sig_logN = np.sqrt(sig_N**2*(np.log(10)/N)**2)
    return logN, sig_logN

def all_column_calc(logDs,unc_Ds):
    D, sig_D = D_column_calc(logDs,unc_Ds)
    N, sig_N = N_column_calc(D, sig_D)
    logN, sig_logN = logN_column_calc(N,sig_N)
    return logN, sig_logN

def calc_fracs(array):
    tot = np.sum(10**array)
    return 10**array/tot

def all_column_calc(logDs, unc_Ds):
    logDs = np.array(logDs)
    unc_Ds = np.array(unc_Ds)
    fracs = calc_fracs(logDs)
    logD = np.log10(np.sum(10**logDs)/1.5e-5)
    return logD,np.sqrt(np.sum(unc_Ds**2 * fracs**2))

D_column_calc(np.array([13.52,13.30]),np.array([0.09,0.11]))


n=100000
Ns = np.zeros(n)
for i in range(n):
    if i%1000 == 0:
        print(i,n)
    d1 = np.random.normal(loc=13.52,scale=0.05)
    d2 = np.random.normal(loc=13.30,scale=0.05)
    log_n = np.log10((10**d1 + 10**d2)/1.5e-5)
    Ns[i] = log_n
plt.figure()
plt.hist(Ns)
print(np.mean(Ns),np.std(Ns))

"""
