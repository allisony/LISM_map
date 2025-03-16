import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import pandas as pd
import numpy as np
from lyapy import voigt

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

pc_in_cm = 3.08567758e18
au_in_cm = 1.49597871e13
lya_rest = 1215.67
c_km = 2.99792458e5

lya_rest = 1215.67
ccgs = 2.99792458e10
e=4.8032e-10            # electron charge in esu
mp=1.6726231e-24        # proton mass in grams
me=mp/1836.             # electron mass in grams


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

def total_tau_profile_func(wave_to_fit,h1_col,h1_b,h1_vel,d2h=1.5e-5,which_line='all lyman',wave_cut_off=2.0):

    """
    Given a wavelength array and parameters (H column density, b value, and 
    velocity centroid), computes the Voigt profile of HI and DI Lyman-alpha
    and returns the combined absorption profile.

    """

    ##### ISM absorbers #####

    if which_line == 'h1_d1':

        ## HI ##
   
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'h1',wave_cut_off=wave_cut_off)
        tauh1=np.interp(wave_to_fit,hwave_all,htau_all)

        ## DI ##

        d1_col = np.log10( (10.**h1_col)*d2h )

        dwave_all,dtau_all=tau_profile(d1_col,h1_vel,h1_b/np.sqrt(2.),'d1',wave_cut_off=wave_cut_off)
        taud1=np.interp(wave_to_fit,dwave_all,dtau_all)


        ## Adding the optical depths and creating the observed profile ##

        tot_tau = tauh1 + taud1
        tot_ism = np.exp(-tot_tau)

    elif which_line =='heii':
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'heii',wave_cut_off=wave_cut_off)
        tau_heii=np.interp(wave_to_fit,hwave_all,htau_all)

        tot_ism = np.exp(-tau_heii)


    else:

        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'h1',wave_cut_off=wave_cut_off)
        tauh1=np.interp(wave_to_fit,hwave_all,htau_all)


        d1_col = np.log10( (10.**h1_col)*d2h )

        dwave_all,dtau_all=tau_profile(d1_col,h1_vel,h1_b/np.sqrt(2.),'d1',wave_cut_off=wave_cut_off)
        taud1=np.interp(wave_to_fit,dwave_all,dtau_all)

        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly_beta',wave_cut_off=wave_cut_off)
        tau_lyb=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly_gamma',wave_cut_off=wave_cut_off)
        tau_lyg=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly_delta',wave_cut_off=wave_cut_off)
        tau_lyd=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly_epsilon',wave_cut_off=wave_cut_off)
        tau_lye=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly6',wave_cut_off=wave_cut_off)
        tau_ly6=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly7',wave_cut_off=wave_cut_off)
        tau_ly7=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly8',wave_cut_off=wave_cut_off)
        tau_ly8=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly9',wave_cut_off=wave_cut_off)
        tau_ly9=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly10',wave_cut_off=wave_cut_off)
        tau_ly10=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly11',wave_cut_off=wave_cut_off)
        tau_ly11=np.interp(wave_to_fit,hwave_all,htau_all)
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'ly12',wave_cut_off=wave_cut_off)
        tau_ly12=np.interp(wave_to_fit,hwave_all,htau_all)
        


        tot_tau = tauh1 + taud1 + tau_lyb + tau_lyg + tau_lyd + tau_lye + tau_ly6 + tau_ly7 + tau_ly8 + tau_ly9 + tau_ly10 + tau_ly11 + tau_ly12
        tot_ism = np.exp(-tot_tau)


    return tot_ism


def tau_profile(ncols,vshifts,vdop,which_line,wave_cut_off=2.0):

    """ 
    Computes a Lyman-alpha Voigt profile for HI or DI given column density,
    velocity centroid, and b parameter.

    """

    ## defining rest wavelength, oscillator strength, and damping parameter
    if which_line == 'h1':
        lam0s,fs,gammas=1215.67,0.4161,6.26e8
    elif which_line == 'd1':
        lam0s,fs,gammas=1215.3394,0.4161,6.27e8
    elif which_line == 'mg2_h':
        lam0s,fs,gammas=2796.3543,6.155E-01,2.625E+08
    elif which_line == 'mg2_k':
        lam0s,fs,gammas=2803.5315,3.058E-01,2.595E+08
    elif which_line == 'ly_beta':
        lam0s,fs,gammas=1025.7222,7.914e-2,1.897e8
    elif which_line == 'ly_gamma':
        lam0s,fs,gammas=972.5367,2.901E-02,8.127e7
    elif which_line == 'ly_delta':
        lam0s,fs,gammas=949.7430,1.395E-02,4.204E+07
    elif which_line == 'ly_epsilon':
        lam0s,fs,gammas=937.8034,7.803E-03,2.450E+07
    elif which_line == 'ly6':
        lam0s,fs,gammas=930.7482,4.816E-03,1.450E+07
    elif which_line == 'ly7':
        lam0s,fs,gammas=926.2256,3.185E-03,8.450E+06
    elif which_line == 'ly8':
        lam0s,fs,gammas=923.1503,2.217E-03,4.450E+06
    elif which_line == 'ly9':
        lam0s,fs,gammas=920.9630,1.606E-03,2.450E+06
    elif which_line == 'ly10':
        lam0s,fs,gammas=919.3513,1.201E-03,1.450E+06
    elif which_line == 'ly11':
        lam0s,fs,gammas=918.1293,9.219E-04,8E+05
    elif which_line == 'ly12':
        lam0s,fs,gammas=917.1805,7.231E-04,4e5
    elif which_line == 'heii':
        lam0s,fs,gammas=303.780409484,0.41629,1e8

    else:
        raise ValueError("which_line can only equal 'h1' or 'd1'!")

    Ntot=10.**ncols  # column density of H I gas
    nlam=4000       # number of elements in the wavelength grid
    xsections_onesided=np.zeros(nlam)  # absorption cross sections as a 
                                       # fun<D-O>ction of wavelength (one side of transition)
    u_parameter=np.zeros(nlam)  # Voigt "u" parameter
    nu0s=ccgs/(lam0s*1e-8)  # wavelengths of Lyman alpha in frequency
    nuds=nu0s*vdop/c_km    # delta nus based off vdop parameter
    a_parameter = np.abs(gammas/(4.*np.pi*nuds) ) # Voigt "a" parameter -- damping parameter
    xsections_nearlinecenter = np.sqrt(np.pi)*(e**2)*fs*lam0s/(me*ccgs*vdop*1e13)  # cross-sections 
                                                                           # near Lyman line center

    
    wave_edge=lam0s - wave_cut_off # define wavelength cut off - this is important for the brightest lines and should be increased appropriately.
    wave_symmetrical=np.zeros(2*nlam-1) # huge wavelength array centered around a Lyman transition
    wave_onesided = np.zeros(nlam)  # similar to wave_symmetrical, but not centered 
                                    # around a Lyman transition 
    lamshifts=lam0s*vshifts/c_km  # wavelength shifts from vshifts parameter

    ## find end point for wave_symmetrical array and create wave_symmetrical array
    num_elements = 2*nlam - 1
    first_point = wave_edge
 
    mid_point = lam0s
    end_point = 2*(mid_point - first_point) + first_point
    wave_symmetrical = np.linspace(first_point,end_point,num=num_elements)
    wave_onesided = np.linspace(lam0s,wave_edge,num=nlam)

    freq_onesided = ccgs / (wave_onesided*1e-8)  ## convert "wave_onesided" array to a frequency array

    u_parameter = (freq_onesided-nu0s)/nuds  ## Voigt "u" parameter -- dimensionless frequency offset

    xsections_onesided=xsections_nearlinecenter*voigt.voigt(a_parameter,u_parameter)  ## cross-sections
                                                                                # single sided
                                                                                ## can't do symmetrical 

    xsections_onesided_flipped = xsections_onesided[::-1]
    
    		
    ## making the cross-sections symmetrical
    xsections_symmetrical=np.append(xsections_onesided_flipped[0:nlam-1],xsections_onesided) 
    deltalam=np.max(wave_symmetrical)-np.min(wave_symmetrical)
    dellam=wave_symmetrical[1]-wave_symmetrical[0] 
    nall=np.round(deltalam/dellam)
    wave_all=deltalam*(np.arange(nall)/(nall-1))+wave_symmetrical[0]

    tau_all = np.interp(wave_all,wave_symmetrical+lamshifts,xsections_symmetrical*Ntot)

    return wave_all,tau_all


h1_vel = 0
wave = np.arange(50,1230,0.25)
wave_heii = np.arange(280,320,0.001)
wave_lya = np.arange(913,1230,0.01)
rel_he1 = 0.08
frac_he2 = 0.6

h1_cols = np.array([17.5, 18.0, 18.5, 19.0])

fig = plt.figure(figsize=(14,5))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for i in range(len(h1_cols)):

    atten = total_tau_profile_func_euv(wave,h1_cols[i],h1_vel,rel_he1=rel_he1,frac_he2=frac_he2,which_line='all')
    lya_atten = total_tau_profile_func(wave_lya,h1_cols[i],10.8,0,wave_cut_off=10.,d2h=1.56e-5,which_line='all lyman')
    heii_atten = total_tau_profile_func(wave_heii,np.log10(10**(h1_cols[i])*rel_he1*frac_he2),8,0,wave_cut_off=10.,which_line='heii')
    ax.plot(wave,atten,label=str(h1_cols[i]),color='C'+str(i))
    #ax.plot(wave_lya,lya_atten,color='C'+str(i),label=str(h1_cols[i]))
    #ax.plot(wave_heii,heii_atten,color='m',linewidth=0.5)

    ax2.plot(wave_lya,lya_atten,color='C'+str(i),label=str(h1_cols[i]))


if True:

    hi_lyman_series = np.array([1215.67,1025.7222,972.5367,949.7430,937.8034,930.7482,926.2256,923.1503,920.9630,919.3513,918.1293,917.1805,
                        916.4291, 915.8238, 915.3289, 914.9192, 914.5762,914.2861,914.0385,913.8256,913.6411, 913.4803,913.3391,913.2146,913.1042,913.0059,
                           912.9179,912.8389,912.7676,912.7032])

    heii_lyman_series = hi_lyman_series / 4.

    hei_strong_lines = np.array([584.3339, 537.0293, 522.186, 515.596, 512.07, 509.97, 508.63, 507.71, 507.08, 506.56,506.31,505.90,505.75,505.61])

    hei_autoionization_lines = np.array([205.885, 194.675, 192.257, 191.227])

    tickmark_bottom_hi_lyman = 1.02
    tickmark_top_hi_lyman = 1.2
    tickmark_bottom_heii_lyman = 1.02
    tickmark_top_heii_lyman = 1.2

    for i in range(len(hi_lyman_series)):

        ax.annotate('', xy=(hi_lyman_series[i], tickmark_bottom_hi_lyman), xycoords='data', xytext=(hi_lyman_series[i], tickmark_top_hi_lyman), 
            arrowprops=dict(arrowstyle="-", color='r'))
        ax.annotate('', xy=(heii_lyman_series[i], tickmark_bottom_heii_lyman), xycoords='data', xytext=(heii_lyman_series[i], tickmark_top_heii_lyman), 
            arrowprops=dict(arrowstyle="-", color='b'))

    for i in range(len(hei_strong_lines)):

        ax.annotate('', xy=(hei_strong_lines[i], tickmark_bottom_hi_lyman), xycoords='data', xytext=(hei_strong_lines[i], tickmark_top_hi_lyman), 
            arrowprops=dict(arrowstyle="-", color='g'))

    for i in range(len(hei_autoionization_lines)):

        ax.annotate('', xy=(hei_autoionization_lines[i], tickmark_bottom_hi_lyman), xycoords='data', xytext=(hei_autoionization_lines[i], tickmark_top_hi_lyman), 
            arrowprops=dict(arrowstyle="-", color='m'))


ax.set_xlabel('Wavelength (\AA)',fontsize=18)
ax.set_ylabel('Transmission Fraction', fontsize=18)
ax.legend(loc=4,title='log$_{10}$ N(HI)')

ax2.set_xlabel('Wavelength (\AA)',fontsize=18)

ax.set_yscale('log')
ax.set_ylim([1e-2,1.2])
ax.set_xlim([50,1230])#950])
ax.minorticks_on()

ax2.set_xlim([1214,1217.2])
ax2.minorticks_on()

ax3 = ax2.twiny()
ax3.set_xlim([(1214-1215.67)/1215.67*3e5,(1217.2-1215.67)/1215.67*3e5])
ax3.set_xlabel('Velocity relative to Ly$\\alpha$ rest wavelength (km s$^{-1}$)',fontsize=18)

lw=3
ax.vlines(229,1e-2,1.2,color='gray',linestyle=':',linewidth=lw)
ax.vlines(504,1e-2,1.2,color='gray',linestyle=':',linewidth=lw)
ax.vlines(911,1e-2,1.2,color='gray',linestyle=':',linewidth=lw)

ax.annotate('He II',(229,0.1),rotation=90, fontsize=14, va='bottom', ha='right')
ax.annotate('He I',(504,0.1),rotation=90, fontsize=14, va='bottom', ha='right')
ax.annotate('H I',(911,0.1),rotation=90, fontsize=14, va='bottom', ha='right')

plt.tight_layout()
plt.savefig('EUV_ISM_attenuation.png')
