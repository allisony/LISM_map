from astropy.io.votable import parse_single_table
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import numpy as np
from astroquery.simbad import Simbad
import time
import pandas as pd

plt.ion()


rc('font',**{'family':'sans-serif'})
rc('text', usetex=True)

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

rc('text.latex', preamble=r'\usepackage[helvet]{sfmath}')

 
#customSimbad = Simbad()

#customSimbad.add_votable_fields('flux(U)','flux_bibcode(U)','flux(B)','flux_bibcode(B)','flux(V)','flux_bibcode(V)','flux(J)','flux_bibcode(J)','flux(H)','flux_bibcode(H)','flux(K)', 'flux_bibcode(K)','diameter','fe_h','gj','otype','plx',
#        'plx_error','plx_bibcode','rv_value','rvz_bibcode','sptype','ids','membership')

customSimbad = Simbad()

customSimbad.add_votable_fields(
'diameter',
'fe_h',
'flux(U)',
'flux_bibcode(U)',
'flux_error(U)',
'flux_system(U)',
'flux_unit(U)',
'flux(B)',
'flux_bibcode(B)',
'flux_error(B)',
'flux_system(B)',
'flux_unit(B)',
'flux(V)',
'flux_bibcode(V)',
'flux_error(V)',
'flux_system(V)',
'flux_unit(V)',
'flux(J)',
'flux_bibcode(J)',
'flux_error(J)',
'flux_system(J)',
'flux_unit(J)',
'flux(H)',
'flux_bibcode(H)',
'flux_error(H)',
'flux_system(H)',
'flux_unit(H)',
'flux(K)',
'flux_bibcode(K)',
'flux_error(K)',
'flux_system(K)',
'flux_unit(K)',
'ids',
'membership',
'otype',
'otype(opt)',
'plx',
'plx_bibcode',
'rot',
'rv_value',
'sptype')


#In [72]: customSimbad.list_votable_fields()
#--NOTES--

#1. The parameter filtername must correspond to an existing filter. Filters include: B,V,R,I,J,K.  They are checked by SIMBAD but not astroquery.simbad

#2. Fields beginning with rvz display the data as it is in the database. Fields beginning with rv force the display as a radial velocity. Fields beginning with z force the display as a redshift

#3. For each measurement catalog, the VOTable contains all fields of the first measurement. When applicable, the first measurement is the mean one. 

#Available VOTABLE fields:

#bibcodelist(y1-y2)
#biblio
#cel
#cl.g
#coo(opt)
#coo_bibcode
#coo_err_angle
#coo_err_maja
#coo_err_mina
#coo_qual
#coo_wavelength
#coordinates
#dec(opt)
#dec_prec
#diameter
#dim
#dim_angle
#dim_bibcode
#dim_incl
#dim_majaxis
#dim_minaxis
#dim_qual
#dim_wavelength
#dimensions
#distance
#distance_result
#einstein
#fe_h
#flux(filtername)
#flux_bibcode(filtername)
#flux_error(filtername)
#flux_name(filtername)
#flux_qual(filtername)
#flux_system(filtername)
#flux_unit(filtername)
#fluxdata(filtername)
#gcrv
#gen
#gj
#hbet
#hbet1
#hgam
#id(opt)
#ids
#iras
#irc
#iso
#iue
#jp11
#link_bibcode
#main_id
#measurements
#membership
#mesplx
#mespm
#mk
#morphtype
#mt
#mt_bibcode
#mt_qual
#otype
#otype(opt)
#otypes
#parallax
#plx
#plx_bibcode
#plx_error
#plx_prec
#plx_qual
#pm
#pm_bibcode
#pm_err_angle
#pm_err_maja
#pm_err_mina
#pm_qual
#pmdec
#pmdec_prec
#pmra
#pmra_prec
#pos
#posa
#propermotions
#ra(opt)
#ra_prec
#rot
#rv_value
#rvz_bibcode
#rvz_error
#rvz_qual
#rvz_radvel
#rvz_type
#rvz_wavelength
#sao
#sp
#sp_bibcode
#sp_nature
#sp_qual
#sptype
##td1
#typed_id
#ubv
#uvby
#uvby1
#v*
#velocity
#xmm
#z_value
#For more information on a field:
#Simbad.get_field_description ('field_name') 
#Currently active VOTABLE fields:
# ['main_id', 'coordinates', 'flux(V)', 'flux(K)']
##

## let's do this in chunks for computational simplicity!

colnames=['Gaia ID',
'MAIN_ID',
'RA',
'DEC',
'RA_PREC',
'DEC_PREC',
'COO_ERR_MAJA',
'COO_ERR_MINA',
'COO_ERR_ANGLE',
'COO_QUAL',
'COO_WAVELENGTH',
'COO_BIBCODE',
'Diameter_diameter',
'Diameter_Q',
'Diameter_unit',
'Diameter_error',
'Diameter_filter',
'Diameter_method',
'Diameter_bibcode',
'Fe_H_Teff',
'Fe_H_log_g',
'Fe_H_Fe_H',
'Fe_H_flag',
'Fe_H_CompStar',
'Fe_H_CatNo',
'Fe_H_bibcode',
'FLUX_U',
'FLUX_BIBCODE_U',
'FLUX_ERROR_U',
'FLUX_SYSTEM_U',
'FLUX_UNIT_U',
'FLUX_B',
'FLUX_BIBCODE_B',
'FLUX_ERROR_B',
'FLUX_SYSTEM_B',
'FLUX_UNIT_B',
'FLUX_V',
'FLUX_BIBCODE_V',
'FLUX_ERROR_V',
'FLUX_SYSTEM_V',
'FLUX_UNIT_V',
'FLUX_J',
'FLUX_BIBCODE_J',
'FLUX_ERROR_J',
'FLUX_SYSTEM_J',
'FLUX_UNIT_J',
'FLUX_H',
'FLUX_BIBCODE_H',
'FLUX_ERROR_H',
'FLUX_SYSTEM_H',
'FLUX_UNIT_H',
'FLUX_K',
'FLUX_BIBCODE_K',
'FLUX_ERROR_K',
'FLUX_SYSTEM_K',
'FLUX_UNIT_K',
'IDS',
'MEMBERSHIP',
'OTYPE',
'OTYPE_opt',
'PLX_VALUE',
'PLX_BIBCODE',
'ROT_upVsini',
'ROT_Vsini',
'ROT_err',
'ROT_mes',
'ROT_qual',
'ROT_bibcode',
'RV_VALUE',
'SP_TYPE',
'SP_QUAL',
'SP_BIBCODE',
'SCRIPT_NUMBER_ID']

path = "/Users/aayoungb/Documents/GitHub/LISM_map/targets/"

#distance_range = [30,35]
#plx_min = 1e3/distance_range[1]
#plx_max = 1e3/distance_range[0]
t = pd.read_csv(path+'Stars_with_LyA_to_add.csv')


df = pd.DataFrame(data=None,columns=colnames)

for i,name in enumerate(t['sci_targname'].unique()):
    if i%10==0:
        print(i)
    try:
        r=customSimbad.query_object(name.replace('-',' '),verbose=False)
    except:
        r = -999
    if r is None:
        print(name + ' not in Simbad')
        nan_array = np.empty(len(colnames))
        nan_array[:] = np.nan
        nan_array[0] = i
        nan_array[1] = -999
        df.loc[i] = nan_array
    elif np.isscalar(r): 
        print(name + ' connection terminated, try again later')
        nan_array = np.empty(len(colnames))
        nan_array[:] = np.nan
        nan_array[0] = i
        nan_array[1] = -888
        df.loc[i] = nan_array

    else:
        print('got ' + name)
        row = r.to_pandas()
        df.loc[i] = np.concatenate([np.array([name]),row.values[0]])
  
    time.sleep(0.5) # important for not overloading Simbad


df['sci_targname'] = t['sci_targname'].unique()
#combined = pd.concat([t['sci_targname'].unique(),df],axis=1)

df.to_csv(path + "LyA_stars_to_add_Simbad_properties.csv")

import pdb; pdb.set_trace()

## read in and concatenate them all into one file? Then go in and check the "connection terminated" ones again?


## adding in bright stars that aren't in Gaia but are in Simbad

r = customSimbad.query_criteria("plx>28.58")
df = r.to_pandas()

df_not_in_gaia = pd.DataFrame(data=None,columns=['MAIN_ID'])

j=0
for i in range(len(df)):
    if ('Gaia' not in df['IDS'].loc[i]) and ('Planet' not in df['OTYPE'].loc[i]):
        #print('NO GAIA')
        print(i,df['MAIN_ID'].loc[i],df['IDS'].loc[i],df['OTYPE'].loc[i])
        print(" ")
        df_not_in_gaia.loc[j] = df['MAIN_ID'].loc[i]
        j+=1

df = pd.DataFrame(data=None,columns=colnames)

for i,main_id in enumerate(df_not_in_gaia['MAIN_ID']):
    if i%10==0:
        print(i)
    try:
        r=customSimbad.query_object(main_id,verbose=True)
    except:
        r = -999
    if r is None:
        print(str(main_id) + ' not in Simbad')
        nan_array = np.empty(len(colnames))
        nan_array[:] = np.nan
        nan_array[0] = -999
        nan_array[1] = -999
        df.loc[i] = nan_array
    elif np.isscalar(r): 
        print(str(main_id) + ' connection terminated, try again later')
        nan_array = np.empty(len(colnames))
        nan_array[:] = np.nan
        nan_array[0] = -999
        nan_array[1] = -888
        df.loc[i] = nan_array

    else:
        print('got ' + str(main_id))
        row = r.to_pandas()
        df.loc[i] = np.concatenate([np.array([-999]),row.values[0]])
  
    time.sleep(0.5) # important for not overloading Simbad


df.to_csv(path + "stars_not_in_GaiaDR3.csv")

df_gaia_zeros = pd.DataFrame(data=None,columns = t.colnames)
df_concat=pd.concat([df_gaia_zeros,df],axis=1)
df_concat.to_csv(path + "stars_not_in_GaiaDR3.csv")



## go back to -888 numbers and try again!!


df1 = pd.read_csv(path + "stars_5pc.csv")
df2 = pd.read_csv(path + "stars_5_10pc.csv")
df3 = pd.read_csv(path + "stars_10_15pc.csv")
df4 = pd.read_csv(path + "stars_15_20pc.csv")
df5 = pd.read_csv(path + "stars_20_25pc.csv")
df6 = pd.read_csv(path + "stars_25_30pc.csv")
df5 = pd.read_csv(path + "stars_30_35pc.csv")
df6 = pd.read_csv(path + "stars_not_in_GaiaDR3.csv")

df_big = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
df_big.reset_index(inplace=True) # reindex

df_big.to_csv(path+"stars_0_35pc.csv")


## for missing (-888) numbers:
mask= (df_big['MAIN_ID'] == '-888.0') + (df_big['MAIN_ID'] == '-888')
df_888 = pd.DataFrame(data=None,columns=colnames)

for i in range(mask.sum()):
    name = df_big['DESIGNATION'][mask].values[i]
    try:
        r = customSimbad.query_object(name,verbose=False)
    except:
        r=-999
    if r is None:
        print(str(name) + ' not in Simbad')
        nan_array = np.empty(len(colnames))
        nan_array[:] = np.nan
        nan_array[0] = -999
        nan_array[1] = -999
        df_888.loc[i] = nan_array
    elif np.isscalar(r): 
        print(str(name) + ' connection terminated, try again later')
        nan_array = np.empty(len(colnames))
        nan_array[:] = np.nan
        nan_array[0] = -999
        nan_array[1] = -888
        df_888.loc[i] = nan_array

    else:
        print('got ' + str(name))
        row = r.to_pandas()
        df_888.loc[i] = np.concatenate([np.array([-999]),row.values[0]])
  
    time.sleep(0.5) # important for not overloading Simbad


df_888.to_csv(path+"stars-888.csv") # copy/paste into spreadsheet!



### Masking out <1" binaries:
mask_ruwe = df_big['ruwe'] > 1.2

### Masking out super faint stars/brown dwarfs:
mask_g_mag = df_big['phot_g_mean_mag'] > 15
mask_v_mag = df_big['FLUX_V'] > 14
mask_j_mag = df_big['FLUX_J'] > 10
mask_h_mag = df_big['FLUX_H'] > 10
mask_k_mag = df_big['FLUX_K'] > 10

### mask out stars that aren't in Simbad
mask_noSimbad = df_big['MAIN_ID'] == '-999'

big_mask = mask_ruwe + mask_g_mag + mask_v_mag + mask_j_mag + mask_h_mag + mask_k_mag + mask_noSimbad

df_cut = df_big[~big_mask]
df_cut.reset_index(inplace=True) # reindex

### I'm worried about some of the stars not matching up properly :( need to go back and fix! 

df_cut.to_csv(path+"stars_0_35pc_cut.csv")


df_2RXS = df_cut[['RA','DEC']]
df_2RXS.to_csv(path+"Rosat_2RXS_list.tsv",sep="\t", index=False)

### run through here: http://cdsarc.u-strasbg.fr/cgi-bin/VizieR-3

df_2RXS = pd.read_csv(path+"Rosat_2RXS_result.tsv",sep="\t",skiprows=96,skipfooter=1)
df_2RXS=df_2RXS.drop([0, 1])
df_2RXS.reset_index(inplace=True)
df_2RXS=df_2RXS.drop(['index'],axis=1)

df_2RXS['MAIN_ID'] = 0

Rosat_colnames = list(df_2RXS.columns)

for i in range(len(Rosat_colnames)):

    Rosat_colnames[i] = "ROSAT " + Rosat_colnames[i]

df_cut = df_cut.reindex(columns = list(df_cut.columns) + Rosat_colnames)


for i, RXS1_ID in enumerate(df_2RXS['1RXS']):

    res = list(filter(lambda x: '1RXS ' + RXS1_ID in float(x), df_cut['IDS'])) 

    print(i)

    if len(res) > 0:

        index = np.where(df_cut['IDS'] == res[0])[0][0]
        #print(index)

        df_2RXS['MAIN_ID'].loc[i] = df_cut['MAIN_ID'].loc[index]

        for r in Rosat_colnames:

            #print(r, r[6:], df_2RXS[r[6:]].loc[i])

            df_cut[r].loc[index] = df_2RXS[r[6:]].loc[i]

        #import pdb; pdb.set_trace()

    else:

        df_2RXS['MAIN_ID'].loc[i] = '-777'


df_2RXS.to_csv(path+"Rosat_2RXS_result.csv")
df_cut.to_csv(path+"stars_0_35pc_cut.csv")







