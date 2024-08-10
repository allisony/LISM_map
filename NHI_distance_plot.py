import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import pandas as pd
import numpy as np
from lmfit import Model
#from mathsUtil import *
from scipy.stats import skewnorm
from scipy.stats.mstats import mquantiles
from scipy.optimize import fsolve
import scipy.stats as STS
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


def estSkewNorm(xin,conf=(0.16,0.5,0.84),Guess=None,Mode='Med',Check=True):
    """
    Used for estimating the parameters of a Skew Normal distribution that matches input mode/median and given confidence interval.
    
    Numerically solves system of equations to give the 3 output parameters of corresponding skew normal. May require multiple guess to result in right solution. Output should give practically exact values, see `Check'.
    
    Returns the skew normal parameters (mu, sigma, alpha) --- see wikipedia on skew normal
    
    WARNING: Code is provided with minimal explanation, and verfication --- use with that in mind.
    
    >---
    
    Parameters
    ----------
    xin : array_like, tuple, or list, with 3 entries
        entries correspond to the data value matching the corresponding index in the `conf' key word. By default first entry indicates the lower bound for the central 68% confidence interval, second entry corresponds to median/mode value depending on keyword `Mode', and the third entry is the upper bound for the central 68% confidence interval.
        
    conf : array_like, tuple, or list, with 3 entries
        indicates the values that the skew normal cumulative probability distribution should give with input `xin'. By default, set to median and central 68% confidence interval. If Mode is `Peak' the median equation is replaced by one corresponding to peak of distribution.
    
    Guess : array_like, tuple or list, with 3 entries
        allows for user to input starting point for numerical equation solve. Default values are a general guess. If output does not converge, use Guess to change starting point for computation. May require iteration for adequete solution. Use `Check' to verify. If there is difficult, input parameters may not be well suited for approximation with skew normal.
    
    Mode : str one of ['Peak','Med2','Med','SF']
        Defines to set of equations used in defining the skew normal distribution. If 'Peak' system sets second entry to be the mode of skew normal instead of median. All others are for setting the median, but with slightly different numerical implementations. 'Peak' and 'Med' are the recommended modes.
        
    Check : boolean
        By default it is True. Used as verification on output solution, gives printed diagnostics as check. Outputs for converged solutions should be exact if fully successful.
    
    
    Returns
    -------
    
    out : array_like with 3 entries
        gives the (mu, sigma, alpha) parameters that define the skew normal distribution matching inputs
    
    Notes
    -----
    Printed warnings also given from scipy from fsolve to diagnose progress of numerical solution to system of equations
    
    Examples
    --------
    
    ## Note that here we use data from https://github.com/jspineda/stellarprop for illustration ; see also Pineda et al. 2021b
    
    >> trace = np.genfromtxt('../resources/massradius/fractional_01/chains.csv',delimiter=',',names=True)
    >> scatlb, scatmid, scatub = confidenceInterval(trace['scatter'])  # the scatter psoterior distribution is asymetric, these are typically the reported values in literature
    >> print([scatlb,scatmid,scatub])
    [0.02787424918238516, 0.0320051813038165, 0.03692976181631807]
    >> params = estSkewNorm( [scatlb, scatmid, scatub])
    Mode at [0.03121118]
    Median at 0.032005181304171265
    Result gives centeral 68.0% confidence interval: (0.027874249182851436, 0.03692976181636316)
    Peak at [0.03121118] - [0.00333693]  +  [0.00571858]
    >> print(params)
    [0.02771848 0.0065575  1.95731243]
    
    ## Note that Check outputs reported numerically match nearly exactly to inputs, these would be kinda off if iteration needed
    ## In this example alpha ~2, indicating positive skewness, peak (mode) is at 0.031, a little less than median at 0.032   -- see appendix of Pineda et al. 2021b
    
    
    """
    
    xl, x0, xu = xin
    cl,c0,cu = conf
    if Guess is not None:
        p0 = Guess
    else:
        p0 = (x0, (xu-xl)/2., ((xu-x0) - (x0-xl))/ ((xu-xl)/2.)  )
    
    ## if block used to toggle set of equations to solve using scipy fsolve
    if Mode=='Peak':
        print("Setting Peak of Distribution")
        def eq_sys(p):
            mu,sigma,alpha = p
            t = (x0 - mu)/sigma
            return STS.skewnorm.cdf(xl,alpha,mu,sigma) - conf[0],STS.skewnorm.cdf(xu,alpha,mu,sigma) - conf[2],alpha*STS.norm.pdf(alpha*t) - STS.norm.cdf(alpha*t)*t
    elif Mode=='Med2':
        print("Setting Median of Distribution")
        def eq_sys(p):
            mu,sigma,alpha = p
            return np.power(STS.skewnorm.cdf(xin,alpha,mu,sigma) - np.array(conf),2)
    elif Mode == 'SF':
        print("Setting Median of Distribution")
        def eq_sys(p):
            mu,sigma,alpha = p
            return STS.skewnorm.isf(1-np.array(conf),alpha,mu,sigma) - np.array(xin)
    elif Mode == 'Med':
        print("Setting Median of Distribution")
        def eq_sys(p):
            mu,sigma,alpha = p
            return (STS.skewnorm.cdf(xl,alpha,mu,sigma)-cl,STS.skewnorm.cdf(x0,alpha,mu,sigma)-c0,STS.skewnorm.cdf(xu,alpha,mu,sigma)-cu)

    out = fsolve(eq_sys,p0)

    if Check:
        ff = lambda a: STS.norm.pdf(out[2]*a)*out[2] - a*STS.norm.cdf(a*out[2])
        tm = fsolve(ff,0.2*out[2])
        xm = tm*out[1] + out[0]
        print("Mode at {}".format(xm))
        print("Median at {}".format(STS.skewnorm.median(out[2],out[0],out[1])))
        print("Result gives centeral {0}% confidence interval:".format((conf[2]-conf[0])*100),STS.skewnorm.interval(conf[2]-conf[0],out[2],out[0],out[1]))
        print("Peak at {0} - {1}  +  {2} ".format(xm, xm - xl, xu  - xm))

    return out  # out is mu, sigma, alpha


####################################################


## Read in the data -- see format_spreadsheet.py ###
df = pd.read_csv('targets/NHI_data_July2024.csv')
####################################################


fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#mask1 = df['RA'] >0 #(df['RA'] < 180.) & (df['DEC'] > 0)
quadrant1 = (df['RA'] > 180.) & (df['DEC'] > 0)
quadrant2 = (df['RA'] > 180.) & (df['DEC'] <= 0)
quadrant3 = (df['RA'] <= 180.) & (df['DEC'] > 0)
quadrant4 = (df['RA'] <= 180.) & (df['DEC'] <= 0)

#ax.errorbar(df['distance (pc)'][mask1], df['N(HI)'][mask1], xerr=df['distance error'][mask1], yerr=df['N(HI) uncertainty'][mask1], fmt='o', color='dodgerblue', ecolor='k', mec='k')
ax.errorbar(df['distance (pc)'][quadrant1], df['N(HI)'][quadrant1], xerr=df['distance error'][quadrant1], yerr=df['N(HI) uncertainty'][quadrant1], fmt='o', color='C0', ecolor='k', mec='k',label='NE')
ax.errorbar(df['distance (pc)'][quadrant2], df['N(HI)'][quadrant2], xerr=df['distance error'][quadrant2], yerr=df['N(HI) uncertainty'][quadrant2], fmt='o', color='C1', ecolor='k', mec='k',label='SE')
ax.errorbar(df['distance (pc)'][quadrant3], df['N(HI)'][quadrant3], xerr=df['distance error'][quadrant3], yerr=df['N(HI) uncertainty'][quadrant3], fmt='o', color='C2', ecolor='k', mec='k',label='NW')
ax.errorbar(df['distance (pc)'][quadrant4], df['N(HI)'][quadrant4], xerr=df['distance error'][quadrant4], yerr=df['N(HI) uncertainty'][quadrant4], fmt='o', color='C3', ecolor='k', mec='k',label='SW')


d1 = df['distance (pc)'] <= 5.
d2 = (df['distance (pc)'] > 5.) & (df['distance (pc)'] <= 10.)
d3 = (df['distance (pc)'] > 10.) & (df['distance (pc)'] <= 50.)
d4 = (df['distance (pc)'] > 50.) & (df['distance (pc)'] <= 100.)

lw=5
a=0.5

ax.hlines(np.mean(df['N(HI)'][quadrant1 & d1]),0,5, color='C0',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant1 & d2]),5,10, color='C0',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant1 & d3]),10,50, color='C0',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant1 & d4]),50,100, color='C0',alpha=a,linewidth=lw)

ax.hlines(np.mean(df['N(HI)'][quadrant2 & d1]),0,5, color='C1',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant2 & d2]),5,10, color='C1',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant2 & d3]),10,50, color='C1',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant2 & d4]),50,100, color='C1',alpha=a,linewidth=lw)

ax.hlines(np.mean(df['N(HI)'][quadrant3 & d1]),0,5, color='C2',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant3 & d2]),5,10, color='C2',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant3 & d3]),10,50, color='C2',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant3 & d4]),50,100, color='C2',alpha=a,linewidth=lw)

ax.hlines(np.mean(df['N(HI)'][quadrant4 & d1]),0,5, color='C3',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant4 & d2]),5,10, color='C3',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant4 & d3]),10,50, color='C3',alpha=a,linewidth=lw)
ax.hlines(np.mean(df['N(HI)'][quadrant4 & d4]),50,100, color='C3',alpha=a,linewidth=lw)

print(np.mean(df['N(HI)']), np.std(df['N(HI)']))

print(np.mean(df['N(HI)'][quadrant1]), np.std(df['N(HI)'][quadrant1]))
print(np.mean(df['N(HI)'][quadrant2]), np.std(df['N(HI)'][quadrant2]))
print(np.mean(df['N(HI)'][quadrant3]), np.std(df['N(HI)'][quadrant3]))
print(np.mean(df['N(HI)'][quadrant4]), np.std(df['N(HI)'][quadrant4]))

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
ax.set_ylim([17,19.7])




## distance shell averages? (add in area-weighting later! or not) # 3/10/24 - add in quadrants??
nHI = 10**(df['N(HI)']) / (df['distance (pc)'] * 3.09e18)

d_avgs = np.logspace(0,2,15)
d_bins = np.zeros(len(d_avgs)-1)
for i in range(len(d_avgs)-1):
    d_bins[i] = np.mean([d_avgs[i], d_avgs[i+1]])
nHI_avgs = np.zeros(len(d_bins))
nHI_68p_low = np.copy(d_bins)
nHI_68p_high = np.copy(d_bins)

df_new = pd.DataFrame()

for i in range(len(d_bins)):

    if i == 0:
        mask = df['distance (pc)'] <= d_avgs[i+1]
    else:
        mask = (df['distance (pc)'] > d_avgs[i] ) & (df['distance (pc)'] <= d_avgs[i+1])
        
    nHI_avgs[i] = np.median(nHI[mask])
    nHI_68p_low[i], nHI_68p_high[i] = np.percentile(nHI[mask],[15.9, 84.1])
    new_col = pd.Series(nHI[mask],name=str(i))
    df_new = pd.concat([df_new, new_col], axis=1)

ax2.step(d_bins,nHI_avgs,where='mid', color='k')
ax2.errorbar(d_bins, nHI_avgs, yerr=[nHI_avgs-nHI_68p_low,nHI_68p_high-nHI_avgs], fmt='none', ecolor='k') 
ax2.set_xlabel('Distance (pc)', fontsize=fs)
ax2.set_ylabel('n(HI) (cm$^{-3}$)', fontsize=fs)
ax2.set_xscale('log')

np.savetxt('nHI_averages_with_distance.txt',np.transpose(np.array([d_bins, nHI_avgs, nHI_avgs-nHI_68p_low,nHI_68p_high-nHI_avgs]))) ## Now need to fit with MCMC? Maybe something simpler is warranted, just scipy optimize? Given that we're not dividing up by quadrant. 
###############

fig.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.13, wspace=0.25)

### QUADRANT 1-4

nHI_avgs_1 = np.zeros(len(d_bins))
nHI_68p_low_1 = np.copy(d_bins)
nHI_68p_high_1 = np.copy(d_bins)
for i in range(len(d_bins)):

    if i == 0:
        mask = df['distance (pc)'][quadrant1] <= d_avgs[i+1]
    else:
        mask = (df['distance (pc)'][quadrant1] > d_avgs[i] ) & (df['distance (pc)'][quadrant1] <= d_avgs[i+1])
        
    if len(nHI[quadrant1][mask]) > 0:
        nHI_avgs_1[i] = np.median(nHI[quadrant1][mask])
        nHI_68p_low_1[i], nHI_68p_high_1[i] = np.percentile(nHI[quadrant1][mask],[15.9, 84.1])
    else:
        nHI_avgs_1[i] = np.nan
        nHI_68p_low_1[i], nHI_68p_high_1[i] = [np.nan, np.nan]



nHI_avgs_2 = np.zeros(len(d_bins))
nHI_68p_low_2 = np.copy(d_bins)
nHI_68p_high_2 = np.copy(d_bins)
for i in range(len(d_bins)):

    if i == 0:
        mask = df['distance (pc)'][quadrant2] <= d_avgs[i+1]
    else:
        mask = (df['distance (pc)'][quadrant2] > d_avgs[i] ) & (df['distance (pc)'][quadrant2] <= d_avgs[i+1])
        
    if len(nHI[quadrant2][mask]) > 0:
        nHI_avgs_2[i] = np.median(nHI[quadrant2][mask])
        nHI_68p_low_2[i], nHI_68p_high_2[i] = np.percentile(nHI[quadrant2][mask],[15.9, 84.1])
    else:
        nHI_avgs_2[i] = np.nan
        nHI_68p_low_2[i], nHI_68p_high_2[i] = [np.nan, np.nan]



nHI_avgs_3 = np.zeros(len(d_bins))
nHI_68p_low_3 = np.copy(d_bins)
nHI_68p_high_3 = np.copy(d_bins)
for i in range(len(d_bins)):

    if i == 0:
        mask = df['distance (pc)'][quadrant3] <= d_avgs[i+1]
    else:
        mask = (df['distance (pc)'][quadrant3] > d_avgs[i] ) & (df['distance (pc)'][quadrant3] <= d_avgs[i+1])

    if len(nHI[quadrant3][mask]) > 0:
        nHI_avgs_3[i] = np.median(nHI[quadrant3][mask])
        nHI_68p_low_3[i], nHI_68p_high_3[i] = np.percentile(nHI[quadrant3][mask],[15.9, 84.1])
    else:
        nHI_avgs_3[i] = np.nan
        nHI_68p_low_3[i], nHI_68p_high_3[i] = [np.nan, np.nan]



nHI_avgs_4 = np.zeros(len(d_bins))
nHI_68p_low_4 = np.copy(d_bins)
nHI_68p_high_4 = np.copy(d_bins)
for i in range(len(d_bins)):

    if i == 0:
        mask = df['distance (pc)'][quadrant4] <= d_avgs[i+1]
    else:
        mask = (df['distance (pc)'][quadrant4] > d_avgs[i] ) & (df['distance (pc)'][quadrant4] <= d_avgs[i+1])

    if len(nHI[quadrant4][mask]) > 0:
        nHI_avgs_4[i] = np.median(nHI[quadrant4][mask])
        nHI_68p_low_4[i], nHI_68p_high_4[i] = np.percentile(nHI[quadrant4][mask],[15.9, 84.1])
    else:
        nHI_avgs_4[i] = np.nan
        nHI_68p_low_4[i], nHI_68p_high_4[i] = [np.nan, np.nan]




#####
ax2.plot(d_bins,nHI_avgs_1,'o', color='C0',mec='k',label='NE')
ax2.plot(d_bins,nHI_avgs_2,'o', color='C1',mec='k',label='SE')
ax2.plot(d_bins,nHI_avgs_3,'o', color='C2',mec='k',label='NW')
ax2.plot(d_bins,nHI_avgs_4,'o', color='C3',mec='k',label='SW')

ax2.legend(title="Sky quadrants")

#ax2.errorbar(d_bins, nHI_avgs, yerr=[nHI_avgs-nHI_68p_low,nHI_68p_high-nHI_avgs], fmt='none', ecolor='k')



def radial_model(d_array,max_value, max_distance):

    nhi_array = max_value * max_distance / d_array

    nhi_array[d_array < max_distance] = 0

    return nhi_array

def grab_random_value(x):

    # first drop nans
    x = x[~np.isnan(x)]

    return x[np.random.randint(len(x))]

def generate_random_nhi_array(df,useindices=['0','1','2','3','4','5','6','7','8','9','10','11','12','13']):

    random_nhi_array = np.zeros(len(useindices))

    for i in range(len(useindices)):

        random_nhi_array[i] = grab_random_value(np.array(df[useindices[i]]))

    return random_nhi_array



def fit_model_1_components(d_array, mv1, md1):

    nhi_array1 = radial_model(d_array, mv1, md1)

    return nhi_array1 


model_1_components = Model(fit_model_1_components)


model_1_components.set_param_hint('mv1', value = 0.1,
               min=0.06,max=0.12,vary=True) 
model_1_components.set_param_hint('md1', value = 1.5,
               min=1,max=5,vary=True) 




params = model_1_components.make_params()
model_1_components.print_param_hints()

initial_model_profile_1_components = model_1_components.eval(params, d_array=d_bins)


result_1_components = model_1_components.fit(nHI_avgs[2:], d_array=d_bins[2:])#, weights=1./unc) # weights need fixing!
print(result_1_components.fit_report())




def fit_model_2_components(d_array, mv1, md1, mv2, md2):

    nhi_array1 = radial_model(d_array, mv1, md1)
    nhi_array2 = radial_model(d_array, mv2, md2)

    return nhi_array1 + nhi_array2 


model_2_components = Model(fit_model_2_components)


model_2_components.set_param_hint('mv1', value = 0.1,
               min=0.06,max=0.12,vary=True) 
model_2_components.set_param_hint('md1', value = 1.5,
               min=1,max=3,vary=True) 

model_2_components.set_param_hint('mv2', value = 0.06,
               min=0.02,max=0.10,vary=True) 
model_2_components.set_param_hint('md2', value = 3,
               min=1.5,max=5,vary=True) 




params = model_2_components.make_params()
model_2_components.print_param_hints()

initial_model_profile_2_components = model_2_components.eval(params, d_array=d_bins)


result_2_components = model_2_components.fit(nHI_avgs[2:], d_array=d_bins[2:])#, weights=1./unc) # weights need fixing!
print(result_2_components.fit_report())
fit_2_components = fit_model_2_components(d_bins[2:], result_2_components.best_values['mv1'], result_2_components.best_values['md1'], result_2_components.best_values['mv2'],  result_2_components.best_values['md2'])
#ax2.plot(d_bins[2:],fit_2_components,color='b')




def fit_model_3_components(d_array, mv1, md1, mv2, md2, mv3, md3):

    nhi_array1 = radial_model(d_array, mv1, md1)
    nhi_array2 = radial_model(d_array, mv2, md2)
    nhi_array3 = radial_model(d_array, mv3, md3)

    return nhi_array1 + nhi_array2 + nhi_array3 


model_3_components = Model(fit_model_3_components)


model_3_components.set_param_hint('mv1', value = 0.1,
               min=0.06,max=0.12,vary=True) 
model_3_components.set_param_hint('md1', value = 1.5,
               min=1,max=3,vary=True) 

model_3_components.set_param_hint('mv2', value = 0.06,
               min=0.02,max=0.10,vary=True) 
model_3_components.set_param_hint('md2', value = 3,
               min=1.5,max=5,vary=True) 

model_3_components.set_param_hint('mv3', value = 0.02,
               min=0.0001,max=0.06,vary=True) 
model_3_components.set_param_hint('md3', value = 10.5,
               min=5,max=20,vary=True) 



params = model_3_components.make_params()
model_3_components.print_param_hints()

initial_model_profile_3_components = model_3_components.eval(params, d_array=d_bins)


result_3_components = model_3_components.fit(nHI_avgs[2:], d_array=d_bins[2:])#, weights=1./unc) # weights need fixing!
print(result_3_components.fit_report())

fit_3_components = fit_model_3_components(d_bins[2:], result_2_components.best_values['mv1'], result_2_components.best_values['md1'], result_2_components.best_values['mv2'],  result_2_components.best_values['md2'],result_3_components.best_values['mv3'], result_3_components.best_values['md3'])
#ax2.plot(d_bins[2:],fit_3_components,color='g')





def fit_model_4_components(d_array, mv1, md1, mv2, md2, mv3, md3, mv4, md4):

    nhi_array1 = radial_model(d_array, mv1, md1)
    nhi_array2 = radial_model(d_array, mv2, md2)
    nhi_array3 = radial_model(d_array, mv3, md3)
    nhi_array4 = radial_model(d_array, mv4, md4)

    return nhi_array1 + nhi_array2 + nhi_array3 + nhi_array4


model_4_components = Model(fit_model_4_components)


model_4_components.set_param_hint('mv1', value = 0.1,
               min=0.06,max=0.12,vary=True) 
model_4_components.set_param_hint('md1', value = 1.5,
               min=1,max=3,vary=True) 

model_4_components.set_param_hint('mv2', value = 0.06,
               min=0.02,max=0.10,vary=True) 
model_4_components.set_param_hint('md2', value = 3,
               min=1.5,max=5,vary=True) 

model_4_components.set_param_hint('mv3', value = 0.02,
               min=0.0001,max=0.06,vary=True) 
model_4_components.set_param_hint('md3', value = 10.5,
               min=5,max=20,vary=True) 

model_4_components.set_param_hint('mv4', value = 0.006,
               min=0.00001,max=0.02,vary=True) 
model_4_components.set_param_hint('md4', value = 80,
               min=50,max=100,vary=True) 

params = model_4_components.make_params()
model_4_components.print_param_hints()

initial_model_profile_4_components = model_4_components.eval(params, d_array=d_bins)

unc = np.mean([nHI_avgs-nHI_68p_low,nHI_68p_high-nHI_avgs],axis=0)
unc[1] = unc[0]

result_4_components = model_4_components.fit(nHI_avgs[1:], d_array=d_bins[1:])#, weights=1./unc) # weights need fixing!
print(result_4_components.fit_report())

fit_4_components = fit_model_4_components(d_bins[1:], result_4_components.best_values['mv1'], result_4_components.best_values['md1'], result_4_components.best_values['mv2'],  result_4_components.best_values['md2'], result_4_components.best_values['mv3'], result_4_components.best_values['md3'], result_4_components.best_values['mv4'], result_4_components.best_values['md4'])
ax2.plot(d_bins[1:],fit_4_components,color='r',linewidth=3)

def fit_model_5_components(d_array, mv1, md1, mv2, md2, mv3, md3, mv4, md4, mv5, md5):

    nhi_array1 = radial_model(d_array, mv1, md1)
    nhi_array2 = radial_model(d_array, mv2, md2)
    nhi_array3 = radial_model(d_array, mv3, md3)
    nhi_array4 = radial_model(d_array, mv4, md4)
    nhi_array5 = radial_model(d_array, mv5, md5)


    return nhi_array1 + nhi_array2 + nhi_array3 + nhi_array4 + nhi_array5


model_5_components = Model(fit_model_5_components)


model_5_components.set_param_hint('mv1', value = 0.1,
               min=0.06,max=0.12,vary=True) 
model_5_components.set_param_hint('md1', value = 1.5,
               min=1,max=3,vary=True) 

model_5_components.set_param_hint('mv2', value = 0.06,
               min=0.02,max=0.10,vary=True) 
model_5_components.set_param_hint('md2', value = 3,
               min=1.5,max=5,vary=True) 

model_5_components.set_param_hint('mv3', value = 0.02,
               min=0.0001,max=0.06,vary=True) 
model_5_components.set_param_hint('md3', value = 10.5,
               min=5,max=20,vary=True) 

model_5_components.set_param_hint('mv4', value = 0.006,
               min=0.0001,max=0.04,vary=True) 
model_5_components.set_param_hint('md4', value = 80,
               min=15,max=30,vary=True) 


model_5_components.set_param_hint('mv5', value = 0.006,
               min=0.00001,max=0.02,vary=True) 
model_5_components.set_param_hint('md5', value = 80,
               min=50,max=100,vary=True) 


params = model_5_components.make_params()
model_5_components.print_param_hints()

initial_model_profile_5_components = model_5_components.eval(params, d_array=d_bins)

result_5_components = model_5_components.fit(nHI_avgs[1:], d_array=d_bins[1:])#, weights=1./unc) # weights need fixing!
print(result_5_components.fit_report())

fit_5_components = fit_model_5_components(d_bins[1:], result_5_components.best_values['mv1'], result_5_components.best_values['md1'], result_5_components.best_values['mv2'],  result_5_components.best_values['md2'], result_5_components.best_values['mv3'], result_5_components.best_values['md3'], result_5_components.best_values['mv4'], result_5_components.best_values['md4'], result_5_components.best_values['mv5'], result_5_components.best_values['md5'])
#ax2.plot(d_bins[1:],fit_5_components,color='g',linewidth=3)

import pdb; pdb.set_trace()


d_array = np.logspace(0,2,100)

best_fit_3_components = fit_model_3_components(d_array, **result_3_components.best_values)
#ax2.plot(d_array,best_fit_3_components,'g--',alpha=0.5,label='3 components')

best_fit_4_components = fit_model_4_components(d_array, **result_4_components.best_values)
#ax2.plot(d_array,best_fit_4_components,'r:',alpha=0.5,label='4 components')

best_fit_5_components = fit_model_5_components(d_array, **result_5_components.best_values)
#ax2.plot(d_array,best_fit_5_components,'b-.',alpha=0.5,label='5 components')

ax2.legend()



alphas = np.zeros(len(nHI_avgs[2:]))
locs = alphas.copy()
scales = alphas.copy()
debug=False
for i in range(len(nHI_avgs[2:])):

    
    out = estSkewNorm([nHI_68p_low[i+2],nHI_avgs[i+2],nHI_68p_high[i+2]],conf=(0.159,0.5,0.841),Guess=None,Mode='Peak',Check=False)
    alphas[i] = out[2]
    locs[i] = out[0]
    scales[i] = out[1]

    if debug:
        plt.figure()
        plt.plot(np.arange(0,0.2,0.005),skewnorm.pdf(np.arange(0,0.2,0.0051), out[2], loc=out[0], scale=out[1]))
        plt.title(str(i))
        plt.axvspan(nHI_68p_low[i+2],nHI_68p_low[i+2],color='r',linestyle=':')
        plt.axvspan(nHI_68p_high[i+2],nHI_68p_high[i+2],color='r',linestyle=':')
        plt.axvspan(nHI_avgs[i+2],nHI_avgs[i+2],color='r',linestyle='-')

        print("***")
        print(i, nHI_68p_low[i+2],nHI_avgs[i+2],nHI_68p_high[i+2], out)
        #import pdb; pdb.set_trace()


#sk=skewnorm.pdf(np.arange(0,10,0.01), out[2], loc=out[0], scale=out[1])

nruns=100

mv1_1components = np.zeros(nruns)
md1_1components = np.zeros(nruns)
bic_1components = np.zeros(nruns)


mv1_2components = np.zeros(nruns)
md1_2components = np.zeros(nruns)
mv2_2components = np.zeros(nruns)
md2_2components = np.zeros(nruns)
bic_2components = np.zeros(nruns)


mv1_3components = np.zeros(nruns)
md1_3components = np.zeros(nruns)
mv2_3components = np.zeros(nruns)
md2_3components = np.zeros(nruns)
mv3_3components = np.zeros(nruns)
md3_3components = np.zeros(nruns)
bic_3components = np.zeros(nruns)

mv1_4components = np.zeros(nruns)
md1_4components = np.zeros(nruns)
mv2_4components = np.zeros(nruns)
md2_4components = np.zeros(nruns)
mv3_4components = np.zeros(nruns)
md3_4components = np.zeros(nruns)
mv4_4components = np.zeros(nruns)
md4_4components = np.zeros(nruns)
bic_4components = np.zeros(nruns)


mv1_5components = np.zeros(nruns)
md1_5components = np.zeros(nruns)
mv2_5components = np.zeros(nruns)
md2_5components = np.zeros(nruns)
mv3_5components = np.zeros(nruns)
md3_5components = np.zeros(nruns)
mv4_5components = np.zeros(nruns)
md4_5components = np.zeros(nruns)
mv5_5components = np.zeros(nruns)
md5_5components = np.zeros(nruns)
bic_5components = np.zeros(nruns)


for i in range(nruns):

    if i%10 == 0:
        print(i)

    nHI_array = np.zeros(len(alphas))

    for j in range(len(alphas)):

        nHI_array[j] = skewnorm.rvs(alphas[j], loc=locs[j], scale=scales[j], size=1)


    result_1_components = model_1_components.fit(nHI_array, d_array=d_bins[2:])
    mv1_1components[i] = result_1_components.best_values['mv1']
    md1_1components[i] = result_1_components.best_values['md1']
    bic_1components[i] = result_1_components.bic


    result_2_components = model_2_components.fit(nHI_array, d_array=d_bins[2:])
    mv1_2components[i] = result_2_components.best_values['mv1']
    md1_2components[i] = result_2_components.best_values['md1']
    mv2_2components[i] = result_2_components.best_values['mv2']
    md2_2components[i] = result_2_components.best_values['md2']
    bic_2components[i] = result_2_components.bic


    result_3_components = model_3_components.fit(nHI_array, d_array=d_bins[2:])
    mv1_3components[i] = result_3_components.best_values['mv1']
    md1_3components[i] = result_3_components.best_values['md1']
    mv2_3components[i] = result_3_components.best_values['mv2']
    md2_3components[i] = result_3_components.best_values['md2']
    mv3_3components[i] = result_3_components.best_values['mv3']
    md3_3components[i] = result_3_components.best_values['md3']
    bic_3components[i] = result_3_components.bic
    
    #best_fit_3_components = fit_model_3_components(d_array, **result_3_components.best_values)

    result_4_components = model_4_components.fit(nHI_array, d_array=d_bins[2:])
    mv1_4components[i] = result_4_components.best_values['mv1']
    md1_4components[i] = result_4_components.best_values['md1']
    mv2_4components[i] = result_4_components.best_values['mv2']
    md2_4components[i] = result_4_components.best_values['md2']
    mv3_4components[i] = result_4_components.best_values['mv3']
    md3_4components[i] = result_4_components.best_values['md3']
    mv4_4components[i] = result_4_components.best_values['mv4']
    md4_4components[i] = result_4_components.best_values['md4']
    bic_4components[i] = result_4_components.bic

    #best_fit_4_components = fit_model_4_components(d_array, **result_4_components.best_values)

    result_5_components = model_5_components.fit(nHI_array, d_array=d_bins[2:])
    mv1_5components[i] = result_5_components.best_values['mv1']
    md1_5components[i] = result_5_components.best_values['md1']
    mv2_5components[i] = result_5_components.best_values['mv2']
    md2_5components[i] = result_5_components.best_values['md2']
    mv3_5components[i] = result_5_components.best_values['mv3']
    md3_5components[i] = result_5_components.best_values['md3']
    mv4_5components[i] = result_5_components.best_values['mv4']
    md4_5components[i] = result_5_components.best_values['md4']
    mv5_5components[i] = result_5_components.best_values['mv5']
    md5_5components[i] = result_5_components.best_values['md5']
    bic_5components[i] = result_5_components.bic

    #best_fit_5_components = fit_model_5_components(d_array, **result_5_components.best_values)



### 1 components

fig=plt.figure(figsize=(3,4))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

nbins=100

ax1.hist(mv1_1components,bins=nbins)
ax2.hist(md1_1components,bins=nbins)




### 2 components

fig=plt.figure(figsize=(4,4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


ax1.hist(mv1_2components,bins=nbins)
ax3.hist(md1_2components,bins=nbins)

ax2.hist(mv2_2components,bins=nbins)
ax4.hist(md2_2components,bins=nbins)



### 3 components

fig=plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)


ax1.hist(mv1_3components,bins=nbins)
ax4.hist(md1_3components,bins=nbins)

ax2.hist(mv2_3components,bins=nbins)
ax5.hist(md2_3components,bins=nbins)

ax3.hist(mv3_3components,bins=nbins)
ax6.hist(md3_3components,bins=nbins)

print("n(HI) 1 = ")
print(np.percentile(mv1_3components,[15.9,50,84.1]))
print("d (pc) 1 = ")
print(np.percentile(md1_3components,[15.9,50,84.1]))

print("n(HI) 2 = ")
print(np.percentile(mv2_3components,[15.9,50,84.1]))
print("d (pc) 2 = ")
print(np.percentile(md2_3components,[15.9,50,84.1]))

print("n(HI) 3 = ")
print(np.percentile(mv3_3components,[15.9,50,84.1]))
print("d (pc) 3 = ")
print(np.percentile(md3_3components,[15.9,50,84.1]))


## All sky quadrants, 100,000 samples -- 4/11/2023
#n(HI) 1 = 
#[0.09363765 0.10148347 0.11339648]
#d (pc) 1 = 
#[1.40549396 1.52237173 1.78563521]
#n(HI) 2 = 
#[0.0549109  0.06205843 0.07665225]
#d (pc) 2 = 
#[2.53268226 3.0203544  3.20514459]
#n(HI) 3 = 
#[0.01677562 0.02274926 0.03185993]
#d (pc) 3 = 
#[ 8.59842954 11.94746728 14.36152839]
#
#best_fit_3_components = fit_model_3_components(d_array, np.median(mv1_3components), np.median(md1_3components),
        # np.median(mv2_3components), np.median(md2_3components), 
        # np.median(mv3_3components), np.median(md3_3components))
#mask = d_array >= np.median(md1_3components)
#ax2.plot(d_array[mask],best_fit_3_components[mask],'r--',alpha=0.5,label='3 components')


### 4 components

fig=plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(241)
ax2 = fig.add_subplot(242)
ax3 = fig.add_subplot(243)
ax4 = fig.add_subplot(244)
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax7 = fig.add_subplot(247)
ax8 = fig.add_subplot(248)


ax1.hist(mv1_4components,bins=nbins)
ax5.hist(md1_4components,bins=nbins)

ax2.hist(mv2_4components,bins=nbins)
ax6.hist(md2_4components,bins=nbins)

ax3.hist(mv3_4components,bins=nbins)
ax7.hist(md3_4components,bins=nbins)

ax4.hist(mv4_4components,bins=nbins)
ax8.hist(md4_4components,bins=nbins)


### 5 components

fig=plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(251)
ax2 = fig.add_subplot(252)
ax3 = fig.add_subplot(253)
ax4 = fig.add_subplot(254)
ax5 = fig.add_subplot(255)
ax6 = fig.add_subplot(256)
ax7 = fig.add_subplot(257)
ax8 = fig.add_subplot(258)
ax9 = fig.add_subplot(259)
ax10 = fig.add_subplot(2,5,10)


ax1.hist(mv1_5components,bins=nbins)
ax6.hist(md1_5components,bins=nbins)

ax2.hist(mv2_5components,bins=nbins)
ax7.hist(md2_5components,bins=nbins)

ax3.hist(mv3_5components,bins=nbins)
ax8.hist(md3_5components,bins=nbins)

ax4.hist(mv4_5components,bins=nbins)
ax9.hist(md4_5components,bins=nbins)

ax5.hist(mv5_5components,bins=nbins)
ax10.hist(md5_5components,bins=nbins)



### BIC plots

fig = plt.figure()
ax1 = fig.add_subplot(111)

a=0.2

ax1.hist(bic_1components,bins=nbins,alpha=a,label='1')
ax1.hist(bic_2components,bins=nbins,alpha=a,label='2')
ax1.hist(bic_3components,bins=nbins,alpha=a,label='3')
ax1.hist(bic_4components,bins=nbins,alpha=a,label='4')
ax1.hist(bic_5components,bins=nbins,alpha=a,label='5')

ax1.legend()

## AY 4/11/2023 - 3 component model has the lowest BIC by ~4 for fitting all quadrants
"""
max_value = 0.1
max_distance = 1.5
nhi_array1 = max_value * max_distance/d_array
nhi_array1[d_array < max_distance] = max_value


max_value = 0.06
max_distance = 3
nhi_array2 = max_value * max_distance/d_array
nhi_array2[d_array < max_distance] = 0

max_value = 0.02
max_distance = 10.5
nhi_array3 = max_value * max_distance/d_array
nhi_array3[d_array < max_distance] = 0

max_value = 0.006
max_distance = 80
nhi_array4 = max_value * max_distance/d_array
nhi_array4[d_array < max_distance] = 0


a=0.7

ax2.plot(d_array,nhi_array1,'r:',alpha=a,label='1.5 pc, 17.66 dex')

ax2.plot(d_array,nhi_array2,'r--',alpha=a,label='3 pc, 17.74 dex')
ax2.plot(d_array,nhi_array3,'r-.',alpha=a,label='10.5 pc, 17.81 dex')
ax2.plot(d_array,nhi_array4,'r:',alpha=a,label='80 pc, 18.17 dex')

ax2.plot(d_array,nhi_array1+nhi_array2+nhi_array3+nhi_array4,'r',alpha=a,label='all')
"""



#plt.savefig('NHI_distance.png')
