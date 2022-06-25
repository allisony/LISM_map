import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import pandas as pd
import numpy as np

## Pretty plot setup ###################################
plt.ion()

rc('font',**{'family':'sans-serif'})
rc('text', usetex=True)

label_size = 16
rcParams['xtick.labelsize'] = label_size 
rcParams['ytick.labelsize'] = label_size

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage[helvet]{sfmath}')

fs=18
average_color = 'r'
####################################################


## Read in the data -- see format_spreadsheet.py ###
df = pd.read_csv('NHI_data.csv')
####################################################


fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax.errorbar(df['distance (pc)'], df['N(HI)'], xerr=df['distance error'], yerr=df['N(HI) uncertainty'], fmt='o', color='dodgerblue', ecolor='k', mec='k')

ax.set_xlabel('Distance (pc)', fontsize=fs)
ax.set_ylabel('log$_{10}$[N(HI)/cm$^{-2}$]', fontsize=fs)
ax.set_xscale('log')

ax.minorticks_on()

## n(HI) = 0.1, 0.01 lines
d = np.arange(1,110,10)
ax.plot(d, np.log10(1.0 * d * 3.09e18), linestyle='--', color='gray')
ax.plot(d, np.log10(0.1 * d * 3.09e18), linestyle='--', color='gray')
ax.plot(d, np.log10(0.01 * d * 3.09e18), linestyle='--', color='gray')
ax.plot(d, np.log10(0.001 * d * 3.09e18), linestyle='--', color='gray')

ax.set_xlim([1,100])
ax.set_ylim([17,19.5])




## distance shell averages? (add in area-weighting later! or not)
nHI = 10**(df['N(HI)']) / (df['distance (pc)'] * 3.09e18)

d_avgs = np.logspace(0,2,15)
d_bins = np.zeros(len(d_avgs)-1)
for i in range(len(d_avgs)-1):
    d_bins[i] = np.mean([d_avgs[i], d_avgs[i+1]])
nHI_avgs = np.zeros(len(d_bins))
nHI_68p_low = np.copy(d_bins)
nHI_68p_high = np.copy(d_bins)

for i in range(len(d_bins)):

    if i == 0:
        mask = df['distance (pc)'] <= d_avgs[i+1]
    else:
        mask = (df['distance (pc)'] > d_avgs[i] ) & (df['distance (pc)'] <= d_avgs[i+1])
        
    nHI_avgs[i] = np.median(nHI[mask])
    nHI_68p_low[i], nHI_68p_high[i] = np.percentile(nHI[mask],[15.9, 84.1])

ax2.step(d_bins,nHI_avgs,where='mid', color='k')
ax2.errorbar(d_bins, nHI_avgs, yerr=[nHI_avgs-nHI_68p_low,nHI_68p_high-nHI_avgs], fmt='none', ecolor='k') 
ax2.set_xlabel('Distance (pc)', fontsize=fs)
ax2.set_ylabel('n(HI) (cm$^{-3}$)', fontsize=fs)
ax2.set_xscale('log')
###############

fig.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.13, wspace=0.25)
plt.savefig('NHI_distance.png')
