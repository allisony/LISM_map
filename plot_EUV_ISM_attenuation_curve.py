import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import pandas as pd
import numpy as np

c_km = 2.99792458e5

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



def total_tau_profile_func_euv(wave_to_fit,h1_col,h1_vel,rel_he1=0.08,frac_he2=0.6,which_line='all'):

    """
    Given a wavelength array and parameters (H column density and 
    velocity centroid), computes the attenuation in the EUV.

    """


   
    hwave_all,htau_all=tau_profile_euv(h1_col,h1_vel,which_line='HI',rel_he1=rel_he1,frac_he2=frac_he2)
    tauh1=np.interp(wave_to_fit,hwave_all,htau_all)

    hwave_all2,htau_all2=tau_profile_euv(h1_col,h1_vel,which_line='HeI',rel_he1=rel_he1,frac_he2=frac_he2)
    tauhe1=np.interp(wave_to_fit,hwave_all2,htau_all2)

    hwave_all3,htau_all3=tau_profile_euv(h1_col,h1_vel,which_line='HeII',rel_he1=rel_he1,frac_he2=frac_he2)
    tauhe2=np.interp(wave_to_fit,hwave_all3,htau_all3)


    ## Adding the optical depths and creating the observed profile ##

    if which_line == 'HI':
      tot_tau = tauh1
    elif which_line == 'HeI':
      tot_tau = tauhe1
    elif which_line == 'HeII':
      tot_tau = tauhe2
    elif which_line == 'all':
      tot_tau = tauh1 + tauhe1 + tauhe2
    else:
      print("uh oh something's wrong")
    tot_ism = np.exp(tot_tau)

    return tot_ism

def tau_profile_euv(ncols,vshifts,which_line,rel_he1=0.08,frac_he2=0.6):


    if which_line == 'HI':
        lam0s,Z,lam_min,frac=912.,1,1,1
    elif which_line == 'HeI':
        lam0s,Z,lam_min,frac= 504., 2., 90., rel_he1
    elif which_line == 'HeII':
        lam0s,Z,lam_min,frac= 229., 2., 50., rel_he1*(frac_he2/(1.0-frac_he2))
    else:
        raise ValueError("which_line can only equal 'HI', 'HeI', or 'HeII'!")




    Ntot=10.**ncols  # column density of H I gas

    lamshifts=lam0s*vshifts/c_km  # wavelength shifts from vshifts parameter


    wave_all = np.arange(1,950,0.1)

    #Addition of Gaunt Factors for more accurate EUV attenuation - kf - 08/06-18/17
    gaunt_lam = np.array([9.12,22.8,45.6,91.2,182,304,456,507,570,651,760,912])
    gaunt_fac = np.array([0.515,0.694,0.830,0.939,0.994,0.985,0.942,0.926,0.905,0.878,0.844,0.797])
    gaunt_array = np.interp(wave_all,gaunt_lam,gaunt_fac)

    # Remember that the gaunt factors shift to teh appropriate wavelengths like Z^2, 
    #see section 5-1 of Spitzer(1978)

    gaunt_lam = gaunt_lam/(Z**2)
    gaunt_array = np.interp(wave_all,gaunt_lam,gaunt_fac)
    no_gaunt = wave_all >= lam0s
    gaunt_array[no_gaunt] = 1.0

    # So, by this formulation, tau(lam<912) = ((6.3e-18)*N_tot) * (lam / lam(Lyman Limit))^3
    # in words: optical depth goes like lam^3, so shorter wavelength, smaller optical depth.

    bey = wave_all <= 912. 

    clip_mask = (wave_all >= lam0s) + (wave_all < lam_min)


    #tau = -7.91e-18/(Z**2) * frac * Ntot * (  (wave_all[bey]/(lam0s+lamshifts))**3  )  * gaunt_array[bey]
    tau = -7.91e-18/(Z**2) * frac * Ntot * (  (wave_all/(lam0s+lamshifts))**3  )  * gaunt_array

    #tau_interp = np.interp(wave_all,wave_all[bey],tau)
    tau_interp = tau.copy()
    tau_interp[clip_mask] = 0.0
    tau_interp[~bey] = 0.0

    # He I opacity added,  kf - 06/22/18 ;;;;;

    if which_line == 'HeI':

       txt = np.loadtxt('HeI_Xsec_samson66.txt',delimiter=',')
       he1_wv = txt[:,0]
       he1_xsec = txt[:,1]*1e-18
       he1_interp = np.interp(wave_all,np.flip(he1_wv),np.flip(he1_xsec))

       tau_interp = -he1_interp * frac * Ntot
       tau_interp[clip_mask] = 0.0


    return wave_all,tau_interp

h1_vel = 0
wave = np.arange(50,950,1)

rel_he1 = 0.08
frac_he2 = 0.6

h1_cols = np.array([17.5, 18.0, 18.5])

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(h1_cols)):

    atten = total_tau_profile_func_euv(wave,h1_cols[i],h1_vel,rel_he1=rel_he1,frac_he2=frac_he2,which_line='all')
    ax.plot(wave,atten,label=str(h1_cols[i]))

ax.set_xlabel('Wavelength (\AA)',fontsize=18)
ax.set_ylabel('Transmission Fraction', fontsize=18)
ax.legend(loc=3,title='log$_{10}$ N(HI)')

ax.set_yscale('log')
ax.set_ylim([1e-2,1.1])
ax.set_xlim([50,950])
ax.minorticks_on()

ax.vlines(229,1e-2,1.1,color='gray',linestyle=':')
ax.vlines(504,1e-2,1.1,color='gray',linestyle=':')
ax.vlines(911,1e-2,1.1,color='gray',linestyle=':')

ax.annotate('He II',(229,0.1),rotation=90, fontsize=14, va='bottom', ha='right')
ax.annotate('He I',(504,0.1),rotation=90, fontsize=14, va='bottom', ha='right')
ax.annotate('H I',(911,0.1),rotation=90, fontsize=14, va='bottom', ha='right')

plt.tight_layout()
plt.savefig('EUV_ISM_attenuation.png')
