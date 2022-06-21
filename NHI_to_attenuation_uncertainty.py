from plot_EUV_ISM_attenuation_curve import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import joblib

plt.ion()

from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
rc('xtick', labelsize=16) 
rc('ytick', labelsize=16)

h1_vel = 0


wave = np.arange(100,901,10)
mask = (wave == 100) + (wave == 200) + (wave == 300)

h1_col_mean_array = np.arange(17.,19.6,0.1)
h1_col_sig_array = np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5])

fractional_uncertainties_to_store = np.zeros((len(h1_col_mean_array),len(h1_col_sig_array),3))

####

n = 5000

h1_col_mean=18.5
h1_col_sig=0.3

h1_cols = np.random.normal(loc=h1_col_mean,scale=h1_col_sig,size=n)


rel_he1_val = 0.08
rel_he1_sig = 0.02
frac_he2_val = 0.6
frac_he2_sig = 0.2
he_uncertainty = False




atten_array = np.zeros((n,len(wave)))


fig=plt.figure(figsize=(7,9))
ax=fig.add_subplot(211)
ax1 = fig.add_subplot(212)

for i in range(n):

    if he_uncertainty:
            rel_he1 = np.random.normal(loc=rel_he1_val,scale=rel_he1_sig)
            frac_he2 = np.random.normal(loc=frac_he2_val,scale=frac_he2_sig)
    else:
            rel_he1 = rel_he1_val
            frac_he2 = frac_he2_val

    atten_array[i,:] = total_tau_profile_func_euv(wave,h1_cols[i],h1_vel,rel_he1=rel_he1,frac_he2=frac_he2,which_line='all')

    if i%10 == 0:
        ax.plot(wave,atten_array[i,:], color='k',alpha=0.1,linewidth=0.5)


low1sig, median, high1sig = np.percentile(atten_array,[15.4,50,84.1],axis=0)

avg_1sig = np.mean([high1sig-median, median-low1sig],axis=0)


ax.plot(wave,median,color='deeppink',label='Median with 1-sigma confidence interval')
ax.plot(wave,low1sig,color='deeppink',linestyle='--')
ax.plot(wave,high1sig,color='deeppink',linestyle='--')


#ax.fill_between(wave,low1sig,high1sig,color='deeppink',alpha=0.8)

ax1.set_xlabel('Wavelength (\AA)',fontsize=18)
ax.set_ylabel('Transmission Fraction',fontsize=18)
ax.minorticks_on()
ax.legend()

ax.set_title('log N(HI) = ' + str(h1_col_mean) + ' $\pm$ ' + str(h1_col_sig),fontsize=18)

ax1.plot(wave,avg_1sig/median)
#print(avg_1sig[mask]/median[mask])
ax1.set_ylabel('Fractional Uncertainty',fontsize=18)
ax1.minorticks_on()
ax1.set_yscale('log')
fig.tight_layout()

ax.set_xlim([50,900])
ax1.set_xlim([50,900])

ax1.set_ylim([0.01,10])

ax1.grid(True)




#########

wave = np.arange(100,601,10)
mask = (wave == 100) + (wave == 200) + (wave == 300) + (wave == 400) + (wave == 500) + (wave == 600)

h1_col_mean_array = np.arange(17.4,19.2,0.2)
h1_col_sig_array = np.logspace(-2,-0.3,10)#np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5])

rel_he1_val = 0.08
rel_he1_sig = 0.02
frac_he2_val = 0.6
frac_he2_sig = 0.2
he_uncertainty = False

fractional_uncertainties_to_store = np.zeros((len(h1_col_mean_array),len(h1_col_sig_array),int(mask.sum())))

n=5000


for i in range(len(h1_col_mean_array)):

    print("i = " + str(i) + ' / ' + str(len(h1_col_mean_array)))

    h1_col_mean = h1_col_mean_array[i]

    #plt.figure()

    for j in range(len(h1_col_sig_array)):

        print("j = " + str(j) + ' / ' + str(len(h1_col_sig_array)))

        h1_col_sig = h1_col_sig_array[j]

        h1_cols = np.random.normal(loc=h1_col_mean,scale=h1_col_sig,size=n)


        atten_array = np.zeros((n,len(wave)))

        for k in range(n):

            if he_uncertainty:
                rel_he1 = np.random.normal(loc=rel_he1_val,scale=rel_he1_sig)
                frac_he2 = np.random.normal(loc=frac_he2_val,scale=frac_he2_sig)
            else:
                rel_he1 = rel_he1_val
                frac_he2 = frac_he2_val


            atten_array[k,:] = total_tau_profile_func_euv(wave,h1_cols[k],h1_vel,rel_he1=rel_he1,frac_he2=frac_he2)
            #atten_array[k,:] = total_tau_profile_func_euv(wave,h1_col_mean,h1_vel,rel_he1=rel_he1,frac_he2=frac_he2)

            plt.plot(wave,atten_array[k,:],color='k',alpha=0.5,linewidth=0.7)

        low1sig, median, high1sig = np.percentile(atten_array,[15.4,50,84.1],axis=0)
        avg_1sig = np.mean([high1sig-median, median-low1sig],axis=0)
        frac_uncertainty = avg_1sig/median

        fractional_uncertainties_to_store[i,j,:] = frac_uncertainty[mask]



fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)


#c=ax1.imshow(fractional_uncertainties_to_store[:,:,0].transpose(),norm=LogNorm(vmin=0.001, vmax=1),origin='lower',aspect='auto',
#                              extent=[np.min(h1_col_mean_array),np.max(h1_col_mean_array),np.min(h1_col_sig_array),np.max(h1_col_sig_array)])
#ax2.imshow(fractional_uncertainties_to_store[:,:,1].transpose(),norm=LogNorm(vmin=0.001, vmax=1),origin='lower',aspect='auto',
#                              extent=[np.min(h1_col_mean_array),np.max(h1_col_mean_array),np.min(h1_col_sig_array),np.max(h1_col_sig_array)])
#ax3.imshow(fractional_uncertainties_to_store[:,:,2].transpose(),norm=LogNorm(vmin=0.001, vmax=1),origin='lower',aspect='auto',
#                              extent=[np.min(h1_col_mean_array),np.max(h1_col_mean_array),np.min(h1_col_sig_array),np.max(h1_col_sig_array)])

levels=[0.0001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1e25]

nlevels=20

xx, yy = np.meshgrid(h1_col_mean_array,h1_col_sig_array)
ax1.contourf(xx,yy,fractional_uncertainties_to_store[:,:,0].transpose(),vmin=0.01, vmax=1,levels=levels)
ax2.contourf(xx,yy,fractional_uncertainties_to_store[:,:,1].transpose(),vmin=0.01, vmax=1,levels=levels)
cs=ax3.contourf(xx,yy,fractional_uncertainties_to_store[:,:,2].transpose(),vmin=0.01, vmax=1,levels=levels)
ax4.contourf(xx,yy,fractional_uncertainties_to_store[:,:,3].transpose(),vmin=0.01, vmax=1,levels=levels)
ax5.contourf(xx,yy,fractional_uncertainties_to_store[:,:,4].transpose(),vmin=0.01, vmax=1,levels=levels)
ax6.contourf(xx,yy,fractional_uncertainties_to_store[:,:,5].transpose(),vmin=0.01, vmax=1,levels=levels)

if False:
    ax1.contour(xx,yy,fractional_uncertainties_to_store[:,:,0].transpose(),levels=[0.35],color='k')
    ax2.contour(xx,yy,fractional_uncertainties_to_store[:,:,1].transpose(),levels=[0.35],color='k')
    ax3.contour(xx,yy,fractional_uncertainties_to_store[:,:,2].transpose(),levels=[0.35],color='k')
    ax4.contour(xx,yy,fractional_uncertainties_to_store[:,:,3].transpose(),levels=[0.35],color='k')
    ax5.contour(xx,yy,fractional_uncertainties_to_store[:,:,4].transpose(),levels=[0.35],color='k')
    ax6.contour(xx,yy,fractional_uncertainties_to_store[:,:,5].transpose(),levels=[0.35],color='k')


ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()
ax6.minorticks_on()

ax4.set_xlabel('log$_{10}$ N(HI) (dex)',fontsize=18)
ax5.set_xlabel('log$_{10}$ N(HI) (dex)',fontsize=18)
ax6.set_xlabel('log$_{10}$ N(HI) (dex)',fontsize=18)


ax1.set_ylabel('$\sigma_{N(HI)}$ (dex)',fontsize=18)
ax4.set_ylabel('$\sigma_{N(HI)}$ (dex)',fontsize=18)

ax1.set_title('100 \AA',fontsize=18)
ax2.set_title('200 \AA',fontsize=18)
ax3.set_title('300 \AA',fontsize=18)
ax4.set_title('400 \AA',fontsize=18)
ax5.set_title('500 \AA',fontsize=18)
ax6.set_title('600 \AA',fontsize=18)

ax1.set_xticks([17.5,18.,18.5,19])
ax2.set_xticks([17.5,18.,18.5,19])
ax3.set_xticks([17.5,18.,18.5,19])
ax4.set_xticks([17.5,18.,18.5,19])
ax5.set_xticks([17.5,18.,18.5,19])
ax6.set_xticks([17.5,18.,18.5,19])

fig.subplots_adjust(left=0.1,right=0.82,bottom=0.11,top=0.9)

cax = plt.axes([0.85, 0.11, 0.03, 0.79]) # left, bottom, width, top
cbar = fig.colorbar(cs, cax=cax, ticks=levels,drawedges=True) #also cs.levels
cbar.set_ticklabels([' ', '0.01', '0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1',' '])
cbar.set_label('Fractional uncertainty',fontsize=18)


if he_uncertainty:
    filename = "../fractional_uncertainties_100_600_Ang_with_helium_var.pkl" # save these somewhere else!

else:

    filename = "../fractional_uncertainties_100_600_Ang.pkl"

with open(filename, "wb") as f:

        joblib.dump([h1_col_mean_array,h1_col_sig_array,fractional_uncertainties_to_store], f)

plt.savefig(filename.replace('.pkl','.png'))
