from plot_EUV_ISM_attenuation_curve import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
import joblib
import matplotlib.ticker
import matplotlib.cm as cm

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

he_uncertainty = True


run = False
####

if run:


    #########

    wave = np.arange(100,901,10)
    mask = (wave == 100) + (wave == 200) + (wave == 300) + (wave == 400) + (wave == 500) + (wave == 600) + (wave == 700) + (wave == 800) + (wave == 900)

    h1_col_mean_array = np.arange(17.4,19.2,0.2)
    h1_col_sig_array = np.logspace(-2,-0.3,10)#np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5])

    rel_he1_val = 0.08
    rel_he1_sig = 0.02
    frac_he2_val = 0.6
    frac_he2_sig = 0.2
    

    fractional_uncertainties_to_store = np.zeros((len(h1_col_mean_array),len(h1_col_sig_array),int(mask.sum())))

    n=10000


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
                    #frac_he2 = np.random.normal(loc=frac_he2_val,scale=frac_he2_sig)
                    frac_he2 = np.random.uniform(low=0.0, high=frac_he2_val)
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

else:
    
    if he_uncertainty:
        filename='fractional_uncertainties_100_600_Ang_with_helium_var.pkl'
    else:
        filename = 'fractional_uncertainties_100_600_Ang.pkl'

    with open("../" + filename, "rb") as f:

        h1_col_mean_array,h1_col_sig_array,fractional_uncertainties_to_store = joblib.load(f)


fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(335)
ax6 = fig.add_subplot(336)
ax7 = fig.add_subplot(337)
ax8 = fig.add_subplot(338)
ax9 = fig.add_subplot(339)


#c=ax1.imshow(fractional_uncertainties_to_store[:,:,0].transpose(),norm=LogNorm(vmin=0.001, vmax=vmax),origin='lower',aspect='auto',
#                              extent=[np.min(h1_col_mean_array),np.max(h1_col_mean_array),np.min(h1_col_sig_array),np.max(h1_col_sig_array)])
#ax2.imshow(fractional_uncertainties_to_store[:,:,1].transpose(),norm=LogNorm(vmin=0.001, vmax=vmax),origin='lower',aspect='auto',
#                              extent=[np.min(h1_col_mean_array),np.max(h1_col_mean_array),np.min(h1_col_sig_array),np.max(h1_col_sig_array)])
#ax3.imshow(fractional_uncertainties_to_store[:,:,2].transpose(),norm=LogNorm(vmin=0.001, vmax=vmax),origin='lower',aspect='auto',
#                              extent=[np.min(h1_col_mean_array),np.max(h1_col_mean_array),np.min(h1_col_sig_array),np.max(h1_col_sig_array)])

#levels=[0.0001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,1e35]
levels=[0.0001,0.01,0.05,0.1,0.25,0.5,0.75,1,2,10,100,1e35]#1e35
viridis = cm.get_cmap('viridis')
colors = [viridis(i / (len(levels) - 1)) for i in range(len(levels))]
norm = BoundaryNorm(boundaries=levels + [levels[-1] + 1], ncolors=len(colors))
cmap = ListedColormap(colors)

nlevels=20

vmin=0.001
vmax=100.0

xx, yy = np.meshgrid(h1_col_mean_array,h1_col_sig_array)
ax1.contourf(xx,yy,fractional_uncertainties_to_store[:,:,0].transpose(),levels=levels,norm=norm,cmap=cmap)
ax2.contourf(xx,yy,fractional_uncertainties_to_store[:,:,1].transpose(),levels=levels,norm=norm,cmap=cmap)
cs=ax3.contourf(xx,yy,fractional_uncertainties_to_store[:,:,2].transpose(),levels=levels,norm=norm,cmap=cmap)
ax4.contourf(xx,yy,fractional_uncertainties_to_store[:,:,3].transpose(),levels=levels,norm=norm,cmap=cmap)
ax5.contourf(xx,yy,fractional_uncertainties_to_store[:,:,4].transpose(),levels=levels,norm=norm,cmap=cmap)
ax6.contourf(xx,yy,fractional_uncertainties_to_store[:,:,5].transpose(),levels=levels,norm=norm,cmap=cmap)
ax7.contourf(xx,yy,fractional_uncertainties_to_store[:,:,6].transpose(),levels=levels,norm=norm,cmap=cmap)
ax8.contourf(xx,yy,fractional_uncertainties_to_store[:,:,7].transpose(),levels=levels,norm=norm,cmap=cmap)
ax9.contourf(xx,yy,fractional_uncertainties_to_store[:,:,8].transpose(),levels=levels,norm=norm,cmap=cmap)

if True:
    lw=0.5
    CS1=ax1.contour(xx,yy,fractional_uncertainties_to_store[:,:,0].transpose(),levels=levels,colors='k',linewidths=lw)
    CS2=ax2.contour(xx,yy,fractional_uncertainties_to_store[:,:,1].transpose(),levels=levels,colors='k',linewidths=lw)
    CS3=ax3.contour(xx,yy,fractional_uncertainties_to_store[:,:,2].transpose(),levels=levels,colors='k',linewidths=lw)
    CS4=ax4.contour(xx,yy,fractional_uncertainties_to_store[:,:,3].transpose(),levels=levels,colors='k',linewidths=lw)
    CS5=ax5.contour(xx,yy,fractional_uncertainties_to_store[:,:,4].transpose(),levels=levels,colors='k',linewidths=lw)
    CS6=ax6.contour(xx,yy,fractional_uncertainties_to_store[:,:,5].transpose(),levels=levels,colors='k',linewidths=lw)
    CS7=ax7.contour(xx,yy,fractional_uncertainties_to_store[:,:,6].transpose(),levels=levels,colors='k',linewidths=lw)
    CS8=ax8.contour(xx,yy,fractional_uncertainties_to_store[:,:,7].transpose(),levels=levels,colors='k',linewidths=lw)
    CS9=ax9.contour(xx,yy,fractional_uncertainties_to_store[:,:,8].transpose(),levels=levels,colors='k',linewidths=lw)



ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()
ax6.minorticks_on()
ax7.minorticks_on()
ax8.minorticks_on()
ax9.minorticks_on()

ax7.set_xlabel('log$_{10}$[N(HI)/cm$^{-2}$]',fontsize=18)
ax8.set_xlabel('log$_{10}$[N(HI)/cm$^{-2}$]',fontsize=18)
ax9.set_xlabel('log$_{10}$[N(HI)/cm$^{-2}$]',fontsize=18)


ax1.set_ylabel('$\sigma_{N(HI)}$ (dex)',fontsize=18)
ax4.set_ylabel('$\sigma_{N(HI)}$ (dex)',fontsize=18)
ax7.set_ylabel('$\sigma_{N(HI)}$ (dex)',fontsize=18)

ax1.set_title('100 \AA',fontsize=18)
ax2.set_title('200 \AA',fontsize=18)
ax3.set_title('300 \AA',fontsize=18)
ax4.set_title('400 \AA',fontsize=18)
ax5.set_title('500 \AA',fontsize=18)
ax6.set_title('600 \AA',fontsize=18)
ax7.set_title('700 \AA',fontsize=18)
ax8.set_title('800 \AA',fontsize=18)
ax9.set_title('900 \AA',fontsize=18)

ax1.set_xticks([17.5,18.,18.5,19])
ax2.set_xticks([17.5,18.,18.5,19])
ax3.set_xticks([17.5,18.,18.5,19])
ax4.set_xticks([17.5,18.,18.5,19])
ax5.set_xticks([17.5,18.,18.5,19])
ax6.set_xticks([17.5,18.,18.5,19])
ax7.set_xticks([17.5,18.,18.5,19])
ax8.set_xticks([17.5,18.,18.5,19])
ax9.set_xticks([17.5,18.,18.5,19])

fmt = matplotlib.ticker.FormatStrFormatter('%.2f')
ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10,fmt=fmt)
ax2.clabel(CS2, CS2.levels, inline=True, fontsize=10,fmt=fmt)
ax3.clabel(CS3, CS3.levels, inline=True, fontsize=10,fmt=fmt)
ax4.clabel(CS4, CS4.levels, inline=True, fontsize=10,fmt=fmt)
ax5.clabel(CS5, CS5.levels, inline=True, fontsize=10,fmt=fmt)
ax6.clabel(CS6, CS6.levels, inline=True, fontsize=10,fmt=fmt)
ax7.clabel(CS7, CS7.levels, inline=True, fontsize=10,fmt=fmt)
ax8.clabel(CS8, CS8.levels, inline=True, fontsize=10,fmt=fmt)
ax9.clabel(CS9, CS9.levels, inline=True, fontsize=10,fmt=fmt)


fig.subplots_adjust(left=0.1,right=0.82,bottom=0.11,top=0.9, hspace=0.3)

cax = plt.axes([0.85, 0.11, 0.03, 0.79]) # left, bottom, width, top
cbar = fig.colorbar(cs, cax=cax, ticks=levels,drawedges=True) #also cs.levels
#cbar.set_ticklabels([' ', '0.01', '0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','10',' '])
cbar.set_ticklabels([' ','0.01','0.05', '0.1','0.25','0.5','0.75','1','2','10','100','$>$100'])
cbar.set_label('Fractional uncertainty',fontsize=18)


if he_uncertainty:
    filename = "../fractional_uncertainties_100_600_Ang_with_helium_var.pkl" # save these somewhere else!

else:

    filename = "../fractional_uncertainties_100_600_Ang.pkl"


if run: 
    with open(filename, "wb") as f:
        joblib.dump([h1_col_mean_array,h1_col_sig_array,fractional_uncertainties_to_store], f)

plt.savefig(filename.replace('.pkl','.png'))
