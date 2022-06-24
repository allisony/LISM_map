import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

plt.ion()

from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
rc('xtick', labelsize=16) 
rc('ytick', labelsize=16)

cbar_h=0.075
cbar_w=0.2
cbar_bottom=0.1


file_prefixes = ['10pc','10_20pc','20_30pc','30_50pc','50_100pc','all_outside_10pc','all']
params = ['amp', 'avg', 'scale']

phi = np.linspace(0, 2.*np.pi, 200)
theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 200)
phi_grid, theta_grid=np.meshgrid(phi,theta)

if False:
 for i in range(len(file_prefixes)):

    df = pd.read_csv('Bestfit_hyperparameters_' + file_prefixes[i] + '.csv')

    stars = np.loadtxt('NHI_column_fitted_stars_' + file_prefixes[i] + '.txt')

    NHI = np.loadtxt('NHI_column_map_' + file_prefixes[i] + '.txt')

    q = NHI.transpose()
    unc = np.mean([q[1]-q[0],q[2]-q[1]],axis=0)

    phi_obs = stars[:,0]
    theta_obs = stars[:,1]
    y_obs = stars[:,2]

    print(file_prefixes[i], len(stars))

    for j in range(len(params)):
        print(params[j],np.exp(df['median'].loc[j]), np.exp(df['84.1%'].loc[j])-np.exp(df['median'].loc[j]), np.exp(df['median'].loc[j])-np.exp(df['15.9%'].loc[j]))


    fig=plt.figure(figsize=(24,6))
    ax1=fig.add_subplot(121,projection='mollweide')
    ax2=fig.add_subplot(122,projection='mollweide')

    im=ax1.pcolor(
        -phi+np.pi,
        theta,
        q[1].reshape((len(phi), len(theta))).T,
        vmin=17,vmax=19)

    ax1.scatter(
        -phi_obs+np.pi,
        theta_obs,
        c=y_obs,
        edgecolor="k",vmin=17,vmax=19,marker='*')

    CS=ax1.contour(-phi+np.pi,
        theta,
        q[1].reshape((len(phi), len(theta))).T,
        levels=[17.0, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18.0,18.1, 18.2,18.3, 18.4,18.5,18.6,18.7,18.8, 18.9,19.0],colors='k')

    ax1.clabel(CS, CS.levels, inline=True, fontsize=10)

    #ax1.set_xticklabels(['','','90','','','0','','','270','',''])
    ax1.set_xticklabels([])
    ax1.text(-90*np.pi/180., 75*np.pi/180., '90$^{\circ}$',ha='center')
    ax1.text(0, 75*np.pi/180., '0$^{\circ}$',ha='center')
    ax1.text(90*np.pi/180., 75*np.pi/180., '270$^{\circ}$',ha='center')
    ax1.text(180*np.pi/180., 75*np.pi/180., '180$^{\circ}$',ha='center')
    ax1.text(-180*np.pi/180., 75*np.pi/180., '180$^{\circ}$',ha='center')


    ax1.set_title(file_prefixes[i])
    ax1.grid(True)

    cax=fig.add_axes([0.135, cbar_bottom, cbar_w, cbar_h])

    fig.colorbar(im, orientation="horizontal",cax=cax,label='log10 N(HI) (cm-2)')

    pred_unc = np.mean([(q[2]-q[1]).reshape((len(phi), len(theta))).T, (q[1]-q[0]).reshape((len(phi), len(theta))).T],axis=0)
    im=ax2.pcolor(
        -(phi-np.pi),
        theta,
        (unc/q[1]).reshape((len(phi), len(theta))).T / np.log(10) ,#pred_unc,
        cmap='gray')


    ax2.plot(
        -(phi_obs-np.pi),
        theta_obs,
        'o',ms=7,mfc='none',mec='m')

    cax2=fig.add_axes([0.41, cbar_bottom, cbar_w, cbar_h])
    fig.colorbar(im, orientation="horizontal",cax=cax2,label='log N(HI) uncertainty (dex)')#label='n(HI) uncertainty (cm-3)')#

    #import pdb; pdb.set_trace()



## figure for paper

fig=plt.figure(figsize=(15,12))
ax1=fig.add_subplot(321,projection='mollweide')
ax2=fig.add_subplot(322,projection='mollweide')
ax3=fig.add_subplot(323,projection='mollweide')
ax4=fig.add_subplot(324,projection='mollweide')
ax5=fig.add_subplot(325,projection='mollweide')
ax6=fig.add_subplot(326,projection='mollweide')

axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]


file_prefixes = ['10pc','10_20pc','20_30pc','30_50pc','50_100pc','all_outside_10pc']
titles = ['$<$10 pc','10-20 pc','20-30 pc','30-50 pc','50-100 pc','10-100 pc']

for i in range(len(file_prefixes)):

    df = pd.read_csv('Bestfit_hyperparameters_' + file_prefixes[i] + '.csv')

    stars = np.loadtxt('NHI_column_fitted_stars_' + file_prefixes[i] + '.txt')

    NHI = np.loadtxt('NHI_column_map_' + file_prefixes[i] + '.txt')

    q = NHI.transpose()
    unc = np.mean([q[1]-q[0],q[2]-q[1]],axis=0)

    phi_obs = stars[:,0]
    theta_obs = stars[:,1]
    y_obs = stars[:,2]

    print(file_prefixes[i], len(stars))

    for j in range(len(params)):
        print(params[j],np.exp(df['median'].loc[j]), np.exp(df['84.1%'].loc[j])-np.exp(df['median'].loc[j]), np.exp(df['median'].loc[j])-np.exp(df['15.9%'].loc[j]))



    im=axes_list[i].pcolor(
        -phi+np.pi,
        theta,
        q[1].reshape((len(phi), len(theta))).T,
        vmin=17.5,vmax=19)

    axes_list[i].scatter(
        -phi_obs+np.pi,
        theta_obs,
        c=y_obs,
        edgecolor="k",vmin=17.5,vmax=19,marker='*')

    CS=axes_list[i].contour(-phi+np.pi,
        theta,
        q[1].reshape((len(phi), len(theta))).T,
        levels=[17.5, 17.6, 17.7, 17.8, 17.9, 18.0,18.1, 18.2,18.3, 18.4,18.5,18.6,18.7,18.8, 18.9,19.0],colors='k')

    axes_list[i].clabel(CS, CS.levels, inline=True, fontsize=10)

    #ax1.set_xticklabels(['','','90','','','0','','','270','',''])
    axes_list[i].set_xticklabels([])
    axes_list[i].text(-90*np.pi/180., 75*np.pi/180., '90$^{\circ}$',ha='center')
    axes_list[i].text(0, 75*np.pi/180., '0$^{\circ}$',ha='center')
    axes_list[i].text(90*np.pi/180., 75*np.pi/180., '270$^{\circ}$',ha='center')
    axes_list[i].text(180*np.pi/180., 75*np.pi/180., '180$^{\circ}$',ha='center')
    axes_list[i].text(-180*np.pi/180., 75*np.pi/180., '180$^{\circ}$',ha='center')


    axes_list[i].set_title(titles[i],fontsize=20)
    axes_list[i].grid(True)

    ## residuals!! 
    ## I want this one to plot the fractional uncertainty!
    
    y_pred = griddata((phi_grid.flatten(),theta_grid.flatten()),q[1],(phi_obs,theta_obs))
    residuals = (y_obs - y_pred)
    print(np.percentile(residuals,[15.9, 50, 84.1]))
    print(np.percentile(np.abs(residuals),[15.9, 50, 84.1]))
    print(np.std(residuals))
    plt.figure()
    plt.plot(np.arange(len(residuals)), residuals,'o')
    plt.title(titles[i])


cax=fig.add_axes([0.11, 0.06, 0.78, 0.02])

cb=fig.colorbar(im, orientation="horizontal",cax=cax)
cb.set_label(label='log$_{10}$ N(HI)',size=20)

#plt.savefig('NHI_map.png')
