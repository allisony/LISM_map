import glob
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.ion()

cloud_list = glob.glob('/Users/aayoungb/MyPapers/Proposals/Missions/ESCAPE/Redfield_ISM_cloud_boundaries_l_b/*.csv')

galactic = False
cmap = cm.get_cmap('plasma', len(cloud_list))

number_mid_pts = 10

projection=ccrs.PlateCarree()

def calc_great_circle_angular_distance(phi_a, phi_b, lam_a,lam_b): ## MUST BE IN RADIANS

    #return np.arccos(np.sin(phi_a) * np.sin(phi_b) + np.cos(phi_a) * np.cos(phi_b))
    return np.arccos(np.sin(lam_a) * np.sin(lam_b) + np.cos(lam_a)*np.cos(lam_b)*np.cos(np.abs(phi_a-phi_b)))

def calc_great_circle_intermediate_point(phi_a, phi_b, lam_a, lam_b, f): ## MUST BE IN RADIANS http://www.movable-type.co.uk/scripts/latlong.html

    delta = calc_great_circle_angular_distance(phi_a, phi_b, lam_a,lam_b)

    a = np.sin( (1.0 - f) * delta) / np.sin(delta)
    b = np.sin(f * delta) / np.sin(delta)

    x = a * np.sin(phi_a) * np.cos(lam_a) + b * np.sin(phi_b) * np.cos(lam_b)
    y = a * np.sin(phi_a) * np.sin(lam_a) + b * np.sin(phi_b) * np.sin(lam_b)
    z = a * np.cos(phi_a) + b * np.cos(phi_b)

    phi_mid = np.arccos(z)
    lam_mid = np.arctan2(y, x)


    #if (-np.pi/4. <= phi_a <= 0) & (-np.pi/4. <= phi_b <= 0):
    #    phi_mid *= -1
    #elif -np.pi/2 <= phi_a <= -np.pi/4.:
    #    phi_mid -= np.pi
#
    #elif phi_a <= -np.pi


    #if phi_a < 0:

    #    print(phi_a, phi_b, lam_a, lam_b)
    #    print(phi_mid, lam_mid)

    if (phi_a < 0) & (phi_b < 0):#(lam_a < 0):
        print(phi_a, phi_b, lam_a, lam_b)
        print(phi_mid, lam_mid)

        phi_mid *= -1
        lam_mid -= np.pi
    #ax.plot(phi_mid*180/np.pi, -lam_mid*180/np.pi,'k^',alpha=0.5)

    return phi_mid, lam_mid

def get_n_midpts(phi_a, phi_b, lam_a, lam_b, n_mid_pts):

    fs = np.linspace(0,1,n_mid_pts)
    fs = np.delete(fs,0)

    mid_pts_array = np.zeros((n_mid_pts-1,2))

    for i in range(n_mid_pts-1):

        phi_mid_i, lam_mid_i = calc_great_circle_intermediate_point(phi_a, phi_b, lam_a, lam_b, fs[i])

        mid_pts_array[i,:] = phi_mid_i, lam_mid_i

    return mid_pts_array

def interpolate_cloud(ra1, dec1, number_mid_pts):

    #ra_new = np.zeros(len(ra1) * (number_mid_pts-2))
    #dec_new = ra_new.copy()
    ra_new = np.array([])
    dec_new = np.array([])

    for i in range(len(ra1)-1):

        mid_pts = get_n_midpts(ra1[i], ra1[i+1], dec1[i], dec1[i+1], number_mid_pts)
        #ax.plot(ra1[i]*180/np.pi, -dec1[i]*180/np.pi,'kx',alpha=0.5)
        #ax.plot(ra1[i+1]*180/np.pi, -dec1[i+1]*180/np.pi,'gx',alpha=0.5)


        ra_new=np.append(ra_new, mid_pts[:,0])
        dec_new=np.append(dec_new, mid_pts[:,1])

        #for k in range(len(mid_pts)):
         #   ra_new[i + k] = mid_pts[k][0]
        #    dec_new[i + k] = mid_pts[k][1]


    return ra_new, dec_new
        



fig = plt.figure()
ax = fig.add_subplot(111, projection=projection)
#ax = fig.add_subplot(111)



for k,fn in enumerate(cloud_list):
  if cloud_list[k].split('/')[-1].replace('_l_b.csv','') == 'Blue':


    df = pd.read_csv(fn)

    cloud_shape_tuple = []

    cloud_shape = np.zeros((len(df),2))


    ra = np.zeros(len(df))
    dec = ra.copy()

    l = ra.copy()
    b = dec.copy()

    for i in range(len(df)):
         

        c = SkyCoord(df['l'][i]*u.deg,-df['b'][i]*u.deg,frame='galactic')

        ra[i] = c.icrs.ra.degree
        dec[i] = c.icrs.dec.degree
        l[i] = c.l.degree
        b[i] = c.b.degree

        #cloud_shape[i,:] = c.icrs.ra.value, c.icrs.dec.value
        if True:
            df_new = pd.DataFrame(data={'ra':ra,'dec':dec})
            df_new.to_csv('/Users/aayoungb/MyPapers/Proposals/Missions/ESCAPE/Redfield_ISM_cloud_boundaries_ra_dec/'+cloud_list[k].split('/')[-1].replace('_l_b.csv','_ra_dec.csv'))




    if galactic:
        ax.plot(df['l'],-1*df['b'],color=cmap(k),label = cloud_list[k].split('/')[-1].replace('_l_b.csv',''),transform=projection)
    #ax.plot(pd.Series(l,name='l'),pd.Series(b,name='b'),color=cmap(k),label = cloud_list[k].split('/')[-1].replace('_l_b.csv',''),transform=projection)
    else:
        ax.plot(ra,dec,color=cmap(k),label = cloud_list[k].split('/')[-1].replace('_l_b.csv','').replace('_',' '),transform=projection, marker='o',markersize=5,linestyle='None')
        
    #ra_new, dec_new = interpolate_cloud(ra*np.pi/180., dec*np.pi/180, number_mid_pts)
    #ra_new = coord.Angle(ra_new, unit=u.radian)
    #dec_new = coord.Angle(dec_new, unit=u.radian)
    #ax.plot(ra_new.deg,dec_new.deg,color=cmap(k),linestyle='--',transform=projection)

    #l_new, b_new = interpolate_cloud(df['l']*np.pi/180., df['b']*np.pi/180., number_mid_pts)

    #l_new = coord.Angle(l_new,unit=u.radian)
    #b_new = coord.Angle(b_new,unit=u.radian)
    #ax.plot(l_new.deg,-b_new.deg,color='k',linestyle='--',transform=projection)

ax.invert_xaxis()

ax.set_xticks([180,135, 90 ,45, 0, -45, -90, -135, -180], crs=projection)
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
ax.set_xticklabels(['180','','90','','0','','270','','180'])
"""
lon_formatter = LongitudeFormatter(number_format='g',
                                       degree_symbol='',
                                       dateline_direction_label=True)
lat_formatter = LatitudeFormatter(number_format='.1f',
                                      degree_symbol='')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
"""
ax.legend()

#gl = ax.gridlines(draw_labels=False)

#ax.set_global()

if galactic:

    ax.set_xlabel('Galactic Longitude (deg)',fontsize=18)
    ax.set_ylabel('Galactic Latitude (deg)',fontsize=18)

else:

    ax.set_xlabel('RA (deg)',fontsize=18)
    ax.set_ylabel('Dec (deg)',fontsize=18)

