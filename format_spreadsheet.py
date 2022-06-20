import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord  # High-level coordinates
import astropy.units as u


## read in my data
df = pd.read_csv('/Users/aayoungb/MyPapers/Proposals/Missions/ESCAPE-2/ISM/ISM column densities - Sheet7.csv')


df = df.drop(df[np.isnan(df['N(HI)'])].index) # drop the ones without direct HI data
df = df.drop(df[df['distance (pc)'] > 40].index) # drop the ones above some distance criterion

df.reset_index(inplace=True) # reindex

df['SkyCoord'] = 0 # setup new column for SkyCoord objects

for i in range(len(df)):

    ra_hms = df['RA'].loc[i]
    dec_dms = df['DEC'].loc[i]

    c = SkyCoord(str(ra_hms) + ' ' + str(dec_dms), frame='icrs', unit=(u.hourangle, u.deg))

    df['RA'].loc[i] = c.ra.deg
    df['DEC'].loc[i] = c.dec.deg


NHI_error = np.array([np.array(i.replace('-','').split(',')).astype(float).mean() for i in df['N(HI) uncertainty']]) ## i'm just taking the average for now...

df['N(HI) uncertainty'] = NHI_error

df_to_save = df[['Star Name', 'distance (pc)', 'distance error', 'N(HI)', 'N(HI) uncertainty', 'N(HI) source', 'RA', 'DEC']].copy()


EUVE = False
if EUVE:
    df_euve = pd.read_csv('/Users/aayoungb/MyPapers/Proposals/Missions/ESCAPE-2/ISM/ISM column densities - EUVE NHI measurements.csv')
    df_euve = df_euve.drop(df_euve[df_euve['PLX_VALUE'] < 10.].index) # drop the ones outside 100 pc
    df_euve.reset_index(inplace=True) # reindex

    df_euve['distance (pc)'] = 1e3/df_euve['PLX_VALUE']
    df_euve['distance error'] = df_euve['PLX_ERROR']/df_euve['PLX_VALUE'] * df_euve['distance (pc)']

    #df_euve['n(HI)'] = df_euve['N(HI)'] / (dist * 3.09e18)
    #df_euve['n(HI) uncertainty'] = 0
    NHI_error = np.array([np.array(i.replace('-','').split(',')).astype(float).mean() for i in df_euve['N(HI) uncertainty']]) ## i'm just taking
    
    df_euve['N(HI) uncertainty'] = np.mean([np.log10(df_euve['N(HI)'] + NHI_error)-np.log10(df_euve['N(HI)']),
                         np.log10(df_euve['N(HI)'])-np.log10(df_euve['N(HI)'] - NHI_error)], axis=0)
    df_euve['N(HI)'] = np.log10(df_euve['N(HI)'])
    df_euve['N(HI) source'] = df_euve['Reference']
    df_euve['Star Name'] = df_euve['Name']
    ## average of any asymmetric error bars! probably want to do this right later!
    #nHI_error = np.sqrt(NHI_error**2 * (np.log(10) * 10**df_euve['N(HI)'] / (dist*3.09e18))**2 + \
    #                  dist_err**2 * (1/(dist*3.09e18)**2  * 10**df_euve['N(HI)'])**2)
    #mask_neg_err = nHI_error/df_euve['n(HI)'] >=1
    #nHI_error[mask_neg_err] = df_euve['n(HI)'][mask_neg_err]
    #df_euve['n(HI) uncertainty'] = nHI_error 


    #phi_obs = np.append(phi_obs, df_euve['RA'].values * np.pi/180.)
    #theta_obs = np.append(theta_obs, df_euve['DEC'].values * np.pi/180.)

    #y_obs = np.append(y_obs, df_euve['n(HI)'].values)
    #yerr = np.append(yerr, df_euve['n(HI) uncertainty'].values)

    #d = np.append(d, dist)

    #skycoords = SkyCoord(ra= phi_obs * u.radian, dec = theta_obs * u.radian)
    #X_obs = np.array(skycoords.cartesian.xyz.T) # shape (100,3) -- for unit vectors

    df_euve_to_save = df_euve[['Star Name', 'distance (pc)', 'distance error', 'N(HI)', 'N(HI) uncertainty', 'N(HI) source', 'RA', 'DEC']].copy()


    df_to_save = pd.concat([df_to_save,df_euve_to_save])


df_to_save.to_csv('NHI_data.csv')
