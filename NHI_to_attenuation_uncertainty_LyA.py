from lyapy import lyapy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
import joblib
import matplotlib.cm as cm
import matplotlib.ticker

plt.ion()

from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
rc('xtick', labelsize=16) 
rc('ytick', labelsize=16)

h1_vel = 0

wave = np.array([1215.0, 1215.34, 1215.68, 1216.02, 1216.36, 1216.7])
velocity = (wave-1215.67)/1215.67*3e5

velocity = np.array([-150.,-81.4,0,81.4, 150, 250])
wave = velocity/3e5*1215.67+1215.67
#wave = np.arange(1213,1220,.3)
#velocity = (wave-1215.67)/1215.67*3e5
#mask = (velocity == -200) + (velocity == -100) + (velocity == -50) + (velocity == 0) + (velocity == 50) + (velocity == 100) + (velocity == 200)
#mask = (wave == 1210) +(wave == 1212) + (wave == 1214) + (wave == 1215) + (wave == 1216) + (wave == 1217) + (wave == 1218) + (wave == 1220)+ (wave == 1221)
mask = np.ones(len(wave),dtype=bool)
#DI is at 1215.3430 Å

h1_col_mean_array = np.arange(17.,19.6,0.1)
h1_col_sig_array = np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5])

d2h_uncertainty = True
h1_b_uncertainty = True

run = True
####

if d2h_uncertainty & h1_b_uncertainty:
    filename = "fractional_uncertainties_LyA_with_d2h_b_var.pkl" # save these somewhere else!

else:

    filename = "fractional_uncertainties_LyA_with_d2h_b_var.pkl"




if run:


    #########

    #wave = np.arange(1210,1230,100)
    #velocity = (wave-1215.67)/1215.67*3e5
    
    #mask = (velocity == -200) + (velocity == -100) + (velocity == -50) + (velocity == 0) + (velocity == 50) + (velocity == 100) + (velocity == 200)
    h1_col_mean_array = np.arange(17.4,19.2,0.2)
    h1_col_sig_array = np.logspace(-2,-0.3,10)#np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5])

    d2h_mean = 1.56e-5
    h1_b_mean = 10.8

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

                if d2h_uncertainty:
                    d2h = np.random.normal(loc=d2h_mean,scale=0.04e-5)
                else:
                    d2h = d2h_mean

                if h1_b_uncertainty:
                    h1_b = np.random.normal(loc=h1_b_mean,scale=1.0)
                else:
                    h1_b = h1_b_mean


                atten_array[k,:] = lyapy.total_tau_profile_func(wave,h1_cols[k],h1_b,h1_vel,d2h)
                #atten_array[k,:] = total_tau_profile_func_euv(wave,h1_col_mean,h1_vel,rel_he1=rel_he1,frac_he2=frac_he2)

                #plt.plot(wave,atten_array[k,:],color='k',alpha=0.5,linewidth=0.7)

            low1sig, median, high1sig = np.nanpercentile(atten_array,[15.4,50,84.1],axis=0)
            avg_1sig = np.mean([high1sig-median, median-low1sig],axis=0)
            frac_uncertainty = avg_1sig/median
            

            fractional_uncertainties_to_store[i,j,:] = frac_uncertainty[mask]
            #print(i,j,frac_uncertainty, frac_uncertainty[mask])

else:
    
    with open(filename, "rb") as f:

        h1_col_mean_array,h1_col_sig_array,fractional_uncertainties_to_store = joblib.load(f)


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

levels=[0.0001,0.01,0.05,0.1,0.25,0.5,0.75,1,2,10,100,1e10]#1e35
#levels=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
#levels = np.linspace(np.nanmin(fractional_uncertainties_to_store), np.nanmax(fractional_uncertainties_to_store), 20)
#nlevels=20

viridis = cm.get_cmap('viridis')
colors = [viridis(i / (len(levels) - 1)) for i in range(len(levels))]
norm = BoundaryNorm(boundaries=levels + [levels[-1] + 1], ncolors=len(colors))
cmap = ListedColormap(colors)

vmin=0.01
vmax=10
#norm = LogNorm()
#norm = BoundaryNorm(boundaries=levels, ncolors=len(levels))
#norm=None
xx, yy = np.meshgrid(h1_col_mean_array,h1_col_sig_array)
ax1.contourf(xx,yy,fractional_uncertainties_to_store[:,:,0].transpose(),levels=levels,norm=norm,cmap=cmap)
ax2.contourf(xx,yy,fractional_uncertainties_to_store[:,:,1].transpose(),levels=levels,norm=norm,cmap=cmap)
cs=ax3.contourf(xx,yy,fractional_uncertainties_to_store[:,:,2].transpose(),levels=levels,norm=norm,cmap=cmap)
ax4.contourf(xx,yy,fractional_uncertainties_to_store[:,:,3].transpose(),levels=levels,norm=norm,cmap=cmap)
ax5.contourf(xx,yy,fractional_uncertainties_to_store[:,:,4].transpose(),levels=levels,norm=norm,cmap=cmap)
ax6.contourf(xx,yy,fractional_uncertainties_to_store[:,:,5].transpose(),levels=levels,norm=norm,cmap=cmap)

lw=0.5
CS1=ax1.contour(xx,yy,fractional_uncertainties_to_store[:,:,0].transpose(),levels=levels,colors='k',linewidths=lw)
CS2=ax2.contour(xx,yy,fractional_uncertainties_to_store[:,:,1].transpose(),levels=levels,colors='k',linewidths=lw)
CS3=ax3.contour(xx,yy,fractional_uncertainties_to_store[:,:,2].transpose(),levels=levels,colors='k',linewidths=lw)
CS4=ax4.contour(xx,yy,fractional_uncertainties_to_store[:,:,3].transpose(),levels=levels,colors='k',linewidths=lw)
CS5=ax5.contour(xx,yy,fractional_uncertainties_to_store[:,:,4].transpose(),levels=levels,colors='k',linewidths=lw)
CS6=ax6.contour(xx,yy,fractional_uncertainties_to_store[:,:,5].transpose(),levels=levels,colors='k',linewidths=lw)



ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()
ax6.minorticks_on()
#ax7.minorticks_on()
#ax8.minorticks_on()
#ax9.minorticks_on()
#
ax4.set_xlabel('log$_{10}$[N(HI)/cm$^{-2}$]',fontsize=18)
ax5.set_xlabel('log$_{10}$[N(HI)/cm$^{-2}$]',fontsize=18)
ax6.set_xlabel('log$_{10}$[N(HI)/cm$^{-2}$]',fontsize=18)


ax1.set_ylabel('$\sigma_{N(HI)}$ (dex)',fontsize=18)
ax4.set_ylabel('$\sigma_{N(HI)}$ (dex)',fontsize=18)
#ax7.set_ylabel('$\sigma_{N(HI)}$ (dex)',fontsize=18)

ax1.set_title('1215.06 \AA\ (-150 km/s)',fontsize=18)
ax2.set_title('1215.34 \AA\ (-81.4 km/s)',fontsize=18)
ax3.set_title('1215.67 \AA\ (0 km/s)',fontsize=18)
ax4.set_title('1216.00 \AA\ (+81.4 km/s)',fontsize=18)
ax5.set_title('1216.28 \AA\ (+150 km/s)',fontsize=18)
ax6.set_title('1216.68 \AA\ (+250 km/s)',fontsize=18)
#ax7.set_title('700 \AA',fontsize=18)
#ax8.set_title('800 \AA',fontsize=18)
#ax9.set_title('900 \AA',fontsize=18)

ax1.set_xticks([17.5,18.,18.5,19])
ax2.set_xticks([17.5,18.,18.5,19])
ax3.set_xticks([17.5,18.,18.5,19])
ax4.set_xticks([17.5,18.,18.5,19])
ax5.set_xticks([17.5,18.,18.5,19])
ax6.set_xticks([17.5,18.,18.5,19])
#ax7.set_xticks([17.5,18.,18.5,19])
#ax8.set_xticks([17.5,18.,18.5,19])
#ax9.set_xticks([17.5,18.,18.5,19])

fmt = matplotlib.ticker.FormatStrFormatter('%.2f')
ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10,fmt=fmt)
ax2.clabel(CS2, CS2.levels, inline=True, fontsize=10,fmt=fmt)
ax3.clabel(CS3, CS3.levels, inline=True, fontsize=10,fmt=fmt)
ax4.clabel(CS4, CS4.levels, inline=True, fontsize=10,fmt=fmt)
ax5.clabel(CS5, CS5.levels, inline=True, fontsize=10,fmt=fmt)
ax6.clabel(CS6, CS6.levels, inline=True, fontsize=10,fmt=fmt)

fig.subplots_adjust(left=0.1,right=0.82,bottom=0.11,top=0.9, hspace=0.3)

cax = plt.axes([0.85, 0.11, 0.03, 0.79]) # left, bottom, width, top
cbar = fig.colorbar(cs, cax=cax, ticks=levels,drawedges=True) #also cs.levels
cbar.set_ticklabels([' ','0.01','0.05', '0.1','0.25','0.5','0.75','1','2','10','100','$>$100'])

cbar.set_label('Fractional uncertainty',fontsize=18)

if run: 
    with open(filename, "wb") as f:
        joblib.dump([h1_col_mean_array,h1_col_sig_array,fractional_uncertainties_to_store], f)

plt.savefig(filename.replace('.pkl','.png'))
