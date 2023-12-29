import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import pandas as pd
import numpy as np
from lmfit import Model
from mathsUtil import *
from scipy.stats import skewnorm

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

mask1 = df['RA'] >0 #(df['RA'] < 180.) & (df['DEC'] > 0)

ax.errorbar(df['distance (pc)'][mask1], df['N(HI)'][mask1], xerr=df['distance error'][mask1], yerr=df['N(HI) uncertainty'][mask1], fmt='o', color='dodgerblue', ecolor='k', mec='k')

print(np.mean(df['N(HI)'][mask1]), np.std(df['N(HI)'][mask1]))

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

result_4_components = model_4_components.fit(nHI_avgs[2:], d_array=d_bins[2:])#, weights=1./unc) # weights need fixing!
print(result_4_components.fit_report())



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

result_5_components = model_5_components.fit(nHI_avgs[2:], d_array=d_bins[2:])#, weights=1./unc) # weights need fixing!
print(result_5_components.fit_report())




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

nruns=10000

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
