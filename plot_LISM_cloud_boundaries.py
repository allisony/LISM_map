import glob
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import pandas as pd
import astropy.units as u

cloud_list = glob.glob('/Users/aayoungb/MyPapers/Proposals/Missions/ESCAPE/Redfield_ISM_cloud_boundaries_l_b/*.csv')


cmap = cm.get_cmap('plasma', len(cloud_list))

number_mid_pts = 10


def calc_great_circle_angular_distance(phi_a, phi_b, lam_a,lam_b): ## MUST BE IN RADIANS

    return np.arccos(np.sin(phi_a) * np.sin(phi_b) + np.cos(phi_a) * np.cos(phi_b))

def calc_great_circle_intermediate_point(phi_a, phi_b, lam_a, lam_b, f): ## MUST BE IN RADIANS

    delta = calc_great_circle_angular_distance(phi_a, phi_b, lam_a,lam_b)

    a = np.sin( (1.0 - f) * delta) / np.sin(delta)
    b = np.sin(f * delta) / np.sin(delta)

    x = a * np.cos(phi_a) * np.cos(lam_a) + b * np.cos(phi_b) * np.cos(lam_b)
    y = a * np.cos(phi_a) * np.sin(lam_a) + b * np.cos(phi_b) * np.sin(lam_b)
    z = a * np.sin(phi_a) + b * np.sin(phi_b)

    phi_mid = np.arctan2(z, np.sqrt(x**2 + y**2))
    lam_mid = np.arctan2(y, x)

    return phi_mid, lam_mid

def get_n_midpts(phi_a, phi_b, lam_a, lam_b, n_mid_pts):

    fs = np.linspace(0,1,n_mid_pts)

    mid_pts_array = np.zeros((n_mid_pts,2))

    for i in range(n_mid_pts):

        phi_mid_i, lam_mid_i = calc_great_circle_intermediate_point(phi_a, phi_b, lam_a, lam_b, fs[i])

        mid_pts_array[i,:] = phi_mid_i, lam_mid_i

    return mid_pts_array

def interpolate_cloud(ra, dec, number_mid_pts):

    ra_new = np.zeros(len(ra) * (number_mid_pts-2))
    dec_new = ra_new.copy()

    for i in range(len(ra)-1):

        mid_pts = get_n_midpts(ra[i], ra[i+1], dec[i], dec[i+1], number_mid_pts)

        for k in range(len(mid_pts)):
            ra_new[i + k] = mid_pts[k][0]
            dec_new[i + k] = mid_pts[k][1]


    return ra_new, dec_new
        

#import pdb; pdb.set_trace()

for k,fn in enumerate(cloud_list):

    df = pd.read_csv(fn)

    cloud_shape_tuple = []

    cloud_shape = np.zeros((len(df),2))

    for i in range(len(df)):
         

        c = SkyCoord(df['l'][i]*u.deg,df['b'][i]*u.deg,frame='galactic')

        #cloud_shape_tuple = (c.icrs.ra.value, c.icrs.dec.value)

        #cloud_shape.append(cloud_shape_tuple)

        cloud_shape[i,:] = c.icrs.ra.value, c.icrs.dec.value


    #if (cloud_shape[-1,0] == cloud_shape[0,0]) & (cloud_shape[-1,1] == cloud_shape[0,1]):
    #    cloud_shape = np.delete(cloud_shape,(-1),axis=0)



    ra = coord.Angle(cloud_shape[:,0],unit=u.deg)
    dec = coord.Angle(cloud_shape[:,1],unit=u.deg)

    #mid_pts = get_n_midpts(ra[0].radian, ra[-2].radian, dec[0].radian, dec[-2].radian, number_mid_pts)

    ra_new, dec_new = interpolate_cloud(ra.radian, dec.radian, number_mid_pts)

    ra_new = coord.Angle(ra_new,unit=u.radian)
    dec_new = coord.Angle(dec_new,unit=u.radian)

    ra_new = ra_new.wrap_at(180.*u.deg)

    for i in range(len(ra_new)):
        cloud_shape_tuple.append((ra_new[i].radian,dec_new[i].radian))

    #for i in range(len(mid_pts)):
    #    cloud_shape_tuple.append((mid_pts[i][0],mid_pts[i][1]))
    #    cloud_shape = np.vstack([cloud_shape,[mid_pts[i][0],mid_pts[i][1]]])


    ax.plot(ra_new.radian,dec_new.radian,color=cmap(k),marker='x')
    #ax.add_patch(mpatches.Polygon(cloud_shape_tuple,facecolor=cmap(k),alpha=0.4))


