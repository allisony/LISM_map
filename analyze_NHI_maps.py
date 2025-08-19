import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import astropy.units as u
import astropy.coordinates as coord
from math import radians, cos, sin, asin, sqrt

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

phi = np.linspace(0, 2.*np.pi, 600)
theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 600)
phi_grid, theta_grid=np.meshgrid(phi,theta)

def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points (specified in radians)
        """
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 1.0
        return c * r # should be in radians

if False: 
    stars = pd.read_csv('targets/NHI_data_August2025.csv') #np.loadtxt('NHI_column_fitted_stars_all.txt')
    RA = stars['RA']
    DEC = stars['DEC']
    ## calculating separations from ALL stars
    Z = np.zeros(phi_grid.shape)
    Z2 = np.zeros(phi_grid.shape) # numbers of stars within 20 degrees?
    for i in range(len(theta)):

        print(i)

        for j in range(len(phi)):


            #print(i,j)


            ## draw a 1x1 deg square around this c`ell, and calculate # stars from latlon in that square
            ## or calculate the haversine distance to all latlon stars, and find how many are within X deg radius
            ## divide that number by pi * X**2 sq deg to get the surface density

            sep_array = np.zeros(len(RA))

            for k in range(len(RA)):

                sep_array[k] = haversine(phi[j], theta[i], RA[k], DEC[k])

            #mask = sep_array <= (radius * np.pi/180.)
            #Z[i,j] = mask.sum() / (np.pi * radius**2)
            Z[i,j] = np.min(sep_array)
            Z2[i,j] = (sep_array <= (20. * np.pi/180.)).sum() ## counting the number of stars within 20 degrees of a particular point


    fig = plt.figure()
    ax = fig.add_subplot(111,projection='mollweide')
    im = ax.pcolor(-phi+np.pi, theta, Z*180./np.pi)
    plt.colorbar(im)
    CS=ax.contour(-phi+np.pi,theta,Z*180./np.pi,
        levels=[20.],colors='k')

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='mollweide')
    im = ax.pcolor(-phi+np.pi, theta, Z2)
    plt.colorbar(im)
    CS=ax.contour(-phi+np.pi,theta,Z2,
        levels=[3.],colors='k')
    ax.clabel(CS, inline=True, fontsize=10)

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

make_fig = True

if make_fig:
    fig=plt.figure(figsize=(15,12))
    ax1=fig.add_subplot(321,projection='mollweide')
    ax2=fig.add_subplot(322,projection='mollweide')
    ax3=fig.add_subplot(323,projection='mollweide')
    ax4=fig.add_subplot(324,projection='mollweide')
    ax5=fig.add_subplot(325,projection='mollweide')
    ax6=fig.add_subplot(326,projection='mollweide')

    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]

    figg=plt.figure(figsize=(15,12))
    axx1=figg.add_subplot(321,projection='mollweide')
    axx2=figg.add_subplot(322,projection='mollweide')
    axx3=figg.add_subplot(323,projection='mollweide')
    axx4=figg.add_subplot(324,projection='mollweide')
    axx5=figg.add_subplot(325,projection='mollweide')
    axx6=figg.add_subplot(326,projection='mollweide')

    axxes_list = [axx1, axx2, axx3, axx4, axx5, axx6]



phi = np.linspace(-np.pi, np.pi, 600)
theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 600)

"""
phi_grid, theta_grid = np.meshgrid(phi, theta)

l_grid = np.zeros_like(phi_grid)
b_grid = np.zeros_like(theta_grid)

c = coord.SkyCoord(ra=phi_grid*u.radian,dec=theta_grid*u.radian,frame='icrs')

l_grid = c.galactic.l
l_grid = l_grid.wrap_at('180d', inplace=False)
l_grid = l_grid.radian
b_grid = c.galactic.b.radian

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.unravel_index(np.argmin(np.abs(array - value)), array.shape)
    return array[idx], idx

def find_nearest2(array1, value1, array2, value2):
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)

    dist1 = np.abs(array1 - value1)
    dist2 =np.abs(array2 - value2)

    idx = np.unravel_index(np.argmin(np.sqrt(dist1**2 + dist2**2)), array1.shape)
    return array1[idx], array2[idx], idx


data_new = np.zeros_like(data)
for ii in range(len(phi)):
    print(ii)
    for jj in range(len(theta)):
        tmp1, tmp2, idx = find_nearest2(l_grid, phi_grid[ii,jj], b_grid, theta_grid[ii,jj])
        #print(idx, data[ii,jj])
        data_new[idx] = data[ii,jj]


for i in range(len(phi)):
    phi_i = phi[i]
    for j in range(len(theta)):

        
        theta_i = theta[i]

        find_nearest(l_grid)

closest_index = np.unravel_index(np.argmin(np.abs(np.sqrt((l_grid - phi_grid[i, j])**2 + (b_grid - theta_grid[i,j])**2))), l_grid.shape)

        diff = np.sqrt((l_grid - phi_grid[i, j])**2 + (theta_grid - theta_grid[i, j])**2)
        
        # Find the index of the minimum difference
        closest_index = np.unravel_index(np.argmin(diff), diff.shape)
        
        # Store the result (coordinates and their closest match)
        closest_matches.append(((i, j), closest_index, l_grid[closest_index], theta_grid[closest_index]))

"""
cloud_path = '/Users/aayoungb/Documents/MyPapers/Proposals/Missions/ESCAPE/Redfield_ISM_cloud_boundaries_ra_dec/'
cloud_name = 'Blue'
df_cloud = pd.read_csv(cloud_path + cloud_name + '_ra_dec.csv')
#fig.text(0.5,0.95,cloud_name.replace('_',''),fontsize=22)

file_path = '/Users/aayoungb/Documents/GitHub/LISM_map/Aug2025/'
file_prefixes = ['all_inside_10pc','10_20pc','20_30pc','30_50pc','50_70pc','70_100pc']
titles = ['$<$10 pc','10-20 pc','20-30 pc','30-50 pc','50-70 pc','70-100 pc']
#params = ['amp', 'amp2', 'avg','scale','scale2']

#import pdb; pdb.set_trace()

df_results = pd.DataFrame(columns=['shell','median','upper','lower','stddev','nstars'])

for i in range(len(file_prefixes)):

    df = pd.read_csv(file_path+'Bestfit_hyperparameters_' + file_prefixes[i] + '_upperlowerlimits.csv')

    stars = np.loadtxt(file_path+'NHI_column_fitted_stars_' + file_prefixes[i] + '_upperlowerlimits.txt')

    NHI = np.loadtxt(file_path+'NHI_column_map_' + file_prefixes[i] + '_upperlowerlimits.txt')

    q = NHI.transpose()
    unc = np.mean([q[1]-q[0],q[2]-q[1]],axis=0)

    figs=plt.figure()
    axs = figs.add_subplot(111)
    axs.hist(q[1,:]-q[0,:],bins=100)
    axs.hist(q[2,:]-q[1,:],bins=100,alpha=0.7)

    phi_obs = coord.Angle(stars[:,0] * u.rad)
    theta_obs = coord.Angle(stars[:,1] * u.rad)
    y_obs = stars[:,2]

    phi_obs = phi_obs.wrap_at('180d', inplace=False)
    

    #phi2 = Angle(phi * u.rad)
    #phi3 = phi2.wrap_at('180d', inplace=False)

    print(file_prefixes[i], len(stars))

    for j in range(len(params)):
        print(params[j],np.exp(df['mean'].loc[j]),np.exp(df['median'].loc[j]), np.exp(df['84.1%'].loc[j])-np.exp(df['median'].loc[j]), np.exp(df['median'].loc[j])-np.exp(df['15.9%'].loc[j]))



    print('avg from map = ',np.mean(q[1]), np.mean(unc), np.percentile(q[1],[15.9,50,84.1]), np.percentile(q[1]-q[0],[15.9,50,84.1]), np.percentile(q[2]-q[1],[15.9,50,84.1]))

    if make_fig:
        im=axes_list[i].pcolor( 
            -phi,
            theta,
            np.roll(q[1].reshape((len(phi), len(theta))).T,int(len(phi)/2),axis=1),
            vmin=17.5,vmax=19, cmap='viridis')

        ## THIS IS A POOR FIX: (shifting the stars to match the underlying map)
        # phi_obs = Angle((stars[:,0]-np.pi) * u.rad)
        # phi_obs = phi_obs.wrap_at('180d', inplace=False)

        imm=axxes_list[i].pcolor(
            -phi,
            theta,
            np.roll(q[1].reshape((len(phi), len(theta))).T,int(len(phi)/2),axis=1) - np.exp(df['median'].loc[1]),
            vmin=-0.8,vmax=0.8, cmap='RdBu')

        axes_list[i].scatter( ### THIS IS CORRECT
            -phi_obs,
            theta_obs,
            c=y_obs,
            edgecolor="k",vmin=17.5,vmax=19,marker='*',s=100,edgecolors='w')

        CS=axes_list[i].contour(-phi,
            theta,
            np.roll(q[1].reshape((len(phi), len(theta))).T,int(len(phi)/2),axis=1),
            levels=[17.5, 17.6, 17.7, 17.8, 17.9, 18.0,18.1, 18.2,18.3, 18.4,18.5,18.6,18.7,18.8, 18.9,19.0],colors='k')

        CS2=axxes_list[i].contour(-phi,
            theta,
            np.roll(q[1].reshape((len(phi), len(theta))).T,int(len(phi)/2),axis=1) - np.exp(df['median'].loc[1]),
            levels=[-0.8, -0.7, -0.6, -0.5, -0.4, -0.3 , -0.2, -0.1, 0, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7, 0.8],colors='k')

        #CS3 = axxes_list[i].contour(-phi,theta,Z*180./np.pi, levels=[20.], colors='m')

        axes_list[i].clabel(CS, CS.levels, inline=True, fontsize=10)
        axxes_list[i].clabel(CS2, CS2.levels, inline=True, fontsize=10)

    ####axes_list[i].plot(-df_cloud['ra']*np.pi/180.,df_cloud['dec']*np.pi/180.,'o',color='gray',markersize=5,linestyle="None")
    #axes_list[i].plot(coord.Angle(-df_cloud['ra']*np.pi/180.,unit='radian').wrap_at(np.pi*u.rad),df_cloud['dec']*np.pi/180.,'o',color='gray',markersize=5,linestyle="None")

    #ax1.set_xticklabels(['','','90','','','0','','','270','',''])
        axes_list[i].set_xticklabels([])
        axes_list[i].text(-90*np.pi/180., 75*np.pi/180., '90$^{\circ}$',ha='center')
        axes_list[i].text(0, 75*np.pi/180., '0$^{\circ}$',ha='center')
        axes_list[i].text(90*np.pi/180., 75*np.pi/180., '270$^{\circ}$',ha='center')
        axes_list[i].text(180*np.pi/180., 75*np.pi/180., '180$^{\circ}$',ha='center')
        axes_list[i].text(-180*np.pi/180., 75*np.pi/180., '180$^{\circ}$',ha='center')


        axes_list[i].set_title(titles[i],fontsize=20)
        axes_list[i].grid(True)
    

        axxes_list[i].set_xticklabels([])
        axxes_list[i].text(-90*np.pi/180., 75*np.pi/180., '90$^{\circ}$',ha='center')
        axxes_list[i].text(0, 75*np.pi/180., '0$^{\circ}$',ha='center')
        axxes_list[i].text(90*np.pi/180., 75*np.pi/180., '270$^{\circ}$',ha='center')
        axxes_list[i].text(180*np.pi/180., 75*np.pi/180., '180$^{\circ}$',ha='center')
        axxes_list[i].text(-180*np.pi/180., 75*np.pi/180., '180$^{\circ}$',ha='center')


        axes_list[i].set_title(titles[i],fontsize=20)
        axes_list[i].grid(True)


    ## residuals!! 
    ## I want this one to plot the fractional uncertainty!
    
    y_pred = griddata((phi_grid.flatten(),theta_grid.flatten()),q[1],(coord.Angle(stars[:,0] * u.rad),theta_obs))
    residuals = (y_obs - y_pred)
    print(np.percentile(residuals,[15.9, 50, 84.1]))
    print(np.percentile(np.abs(residuals),[15.9, 50, 84.1]))
    print(np.std(residuals))
    plt.figure()
    plt.plot(np.arange(len(residuals)), residuals,'o')
    plt.title(titles[i])

    df_results.loc[i] = [titles[i], np.exp(df['median'].loc[1]), np.exp(df['84.1%'].loc[1])-np.exp(df['median'].loc[1]), np.exp(df['median'].loc[1])-np.exp(df['15.9%'].loc[1]), np.std(residuals),   len(stars)]


df_results.to_csv('Table1_results.csv')
if make_fig:
    cax=fig.add_axes([0.11, 0.06, 0.78, 0.02])

    cb=fig.colorbar(im, orientation="horizontal",cax=cax)
    cb.set_label(label='log$_{10}$[N(HI)/cm$^{-2}$]',size=20)


    fig.subplots_adjust(top=0.97,right=0.98,left=0.05)


    cax=figg.add_axes([0.11, 0.06, 0.78, 0.02])

    cb=figg.colorbar(imm, orientation="horizontal",cax=cax)
    cb.set_label(label='log$_{10}$[N(HI)/cm$^{-2}$] - log$_{10}$[N(HI,average)/cm$^{-2}$]',size=20)


    figg.subplots_adjust(top=0.97,right=0.98,left=0.05)

    #plt.savefig(file_path+'NHI_map_upperlowerlimits.png')

    fig.savefig(file_path+'NHI_map.png')
    figg.savefig(file_path+'NHI_diff_map.png')
