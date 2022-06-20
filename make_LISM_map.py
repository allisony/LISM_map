import numpy as np
import matplotlib.pyplot as plt   
from matplotlib import rc, rcParams
import pandas as pd
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
import astropy.coordinates as coord
from astropy.modeling.models import Voigt1D
from scipy.interpolate import griddata
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
from jax.config import config
import jaxopt
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import corner as triangle
from scipy.interpolate import griddata

config.update("jax_enable_x64", True)


pd.options.mode.chained_assignment = None

plt.ion()


#rc('font',**{'family':'sans-serif'})
#rc('text', usetex=True)

#label_size = 16
#rcParams['xtick.labelsize'] = label_size 
#rcParams['ytick.labelsize'] = label_size


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

#rc('text.latex', preamble=r'\usepackage[helvet]{sfmath}')

# Make a spherical grid

class GreatCircleDistance(kernels.stationary.Distance):
    def distance(self, X1, X2):
        if jnp.shape(X1) != (3,) or jnp.shape(X2) != (3,):
            raise ValueError(
                "The great-circle distance is only defined for unit 3-vector"
            )
        return jnp.arctan2(jnp.linalg.norm(jnp.cross(X1, X2)), (X1.T @ X2))


phi = np.linspace(0, 2.*np.pi, 200)
theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 200)
phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
phi_grid = phi_grid.flatten()
theta_grid = theta_grid.flatten()
X_grid = np.vstack(
    (
        np.cos(phi_grid) * np.cos(theta_grid),
        np.sin(phi_grid) * np.cos(theta_grid),
        np.sin(theta_grid),
    )
).T

X_grid_RADec = np.vstack((phi_grid,theta_grid))

## read in my data
df = pd.read_csv('/Users/aayoungb/MyPapers/Proposals/Missions/ESCAPE-2/ISM/ISM column densities - Sheet7.csv')

MgII=False
if MgII:
    zero_nHI_mask = df['n(HI)'] == 0
    df['n(HI)'][zero_nHI_mask] = df['n(MgII)'][zero_nHI_mask] / 3.6e-6 # from Linsky+2019
    df['N(HI)'][zero_nHI_mask] = np.log10(df['n(HI)'][zero_nHI_mask] * df['distance (pc)'] * 3.09e18)
    df['N(HI) uncertainty'][zero_nHI_mask] = '0.35' # dex



df = df.drop(df[np.isnan(df['N(HI)'])].index) # drop the ones without direct HI data
df = df.drop(df[df['distance (pc)'] > 40].index) # drop the ones above some distance criterion

df.reset_index(inplace=True) # reindex

df['SkyCoord'] = 0 # setup new column for SkyCoord objects

for i in range(len(df)):

    ra_hms = df['RA'].loc[i]
    dec_dms = df['DEC'].loc[i]

    c = SkyCoord(str(ra_hms) + ' ' + str(dec_dms), frame='icrs', unit=(u.hourangle, u.deg))

    df['RA'].loc[i] = c.ra.deg
    df['DEC'].loc[i] = c.dec.deg

    #df['SkyCoord'].loc[i] = c

NHI_error = np.array([np.array(i.replace('-','').split(',')).astype(float).mean() for i in df['N(HI) uncertainty']]) ## i'm just taking
## average of any asymmetric error bars! probably want to do this right later!
#nHI_error = np.sqrt(NHI_error**2 * (np.log(10) * 10**df['N(HI)'] / (df['distance (pc)']*3.09e18))**2 + \
#                      df['distance error']**2 * (1/(df['distance (pc)']*3.09e18)**2  * 10**df['N(HI)'])**2)
#mask_neg_err = nHI_error/df['n(HI)'] >=1
#nHI_error[mask_neg_err] = df['n(HI)'][mask_neg_err]
#df['n(HI) uncertainty'] = nHI_error ## there is one going negative - going to cut it off

skycoords = SkyCoord(ra= df['RA'] * u.degree, dec = df['DEC'] * u.degree)
X_obs = np.array(skycoords.cartesian.xyz.T) # shape (100,3) -- for unit vectors
theta_obs = df['DEC'].values * np.pi/180. 
phi_obs  = df['RA'].values * np.pi/180.

y_obs = df['N(HI)'].values
yerr=NHI_error

d = df['distance (pc)']

EUVE = False
if EUVE:
    df_euve = pd.read_csv('/Users/aayoungb/MyPapers/Proposals/Missions/ESCAPE-2/ISM/ISM column densities - EUVE NHI measurements.csv')
    df_euve = df_euve.drop(df_euve[df_euve['PLX_VALUE'] < 10.].index) # drop the ones outside 100 pc
    df_euve.reset_index(inplace=True) # reindex

    dist = 1e3/df_euve['PLX_VALUE']
    dist_err = df_euve['PLX_ERROR']/df_euve['PLX_VALUE'] * dist

    df_euve['n(HI)'] = df_euve['N(HI)'] / (dist * 3.09e18)
    df_euve['n(HI) uncertainty'] = 0
    NHI_error = np.array([np.array(i.replace('-','').split(',')).astype(float).mean() for i in df_euve['N(HI) uncertainty']]) ## i'm just taking
    ## average of any asymmetric error bars! probably want to do this right later!
    nHI_error = np.sqrt(NHI_error**2 * (np.log(10) * 10**df_euve['N(HI)'] / (dist*3.09e18))**2 + \
                      dist_err**2 * (1/(dist*3.09e18)**2  * 10**df_euve['N(HI)'])**2)
    mask_neg_err = nHI_error/df_euve['n(HI)'] >=1
    nHI_error[mask_neg_err] = df_euve['n(HI)'][mask_neg_err]
    df_euve['n(HI) uncertainty'] = nHI_error 


    phi_obs = np.append(phi_obs, df_euve['RA'].values * np.pi/180.)
    theta_obs = np.append(theta_obs, df_euve['DEC'].values * np.pi/180.)

    y_obs = np.append(y_obs, df_euve['n(HI)'].values)
    yerr = np.append(yerr, df_euve['n(HI) uncertainty'].values)

    d = np.append(d, dist)

    skycoords = SkyCoord(ra= phi_obs * u.radian, dec = theta_obs * u.radian)
    X_obs = np.array(skycoords.cartesian.xyz.T) # shape (100,3) -- for unit vectors

# Plot the map
fig=plt.figure()
ax=fig.add_subplot(111,projection='mollweide')
ax.scatter(
    phi_obs-np.pi,
    theta_obs,
    c=y_obs,
    edgecolor="k",vmin=17.0,vmax=19.0)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\theta$")
_ = ax.set_title("data")

plt.figure()
plt.errorbar(d,y_obs,yerr=yerr,fmt='ko')
plt.xlabel('Distance (pc)',fontsize=18)
plt.ylabel('N(HI)',fontsize=18)






def numpyro_model(X_obs, yerr, y=None):
    avg1 = numpyro.sample("log_avg", dist.Cauchy(18,2))

    amp1 = numpyro.sample("log_amp", dist.Cauchy(0,10))
    scale1 = numpyro.sample("log_scale", dist.Normal(-1,0.5))
    #sigma1 = numpyro.sample("sigma", dist.HalfNormal(0.2))
    #c1 = numpyro.sample("log_c",dist.Normal(0,10))

    kernel = jnp.exp(amp1) * kernels.Exp(jnp.exp(scale1), distance=GreatCircleDistance()) #+ kernels.Constant(jnp.exp(c1))

    gp=GaussianProcess(
        kernel,
        X_obs,
        diag=yerr**2, #+ (sigma1 * jnp.log(10) * y)**2, 
        mean=jnp.exp(avg1))


    numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    if y is not None:
        numpyro.deterministic("pred", gp.predict(y, X_grid))


nuts_kernel = NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=2,
    progress_bar=True,
)
rng_key = jax.random.PRNGKey(34913)


mcmc.run(rng_key, X_obs, yerr, y=y_obs)
samples = mcmc.get_samples()
pred = samples["pred"].block_until_ready()  # Blocking to get timing right



data = az.from_numpyro(mcmc)
print(az.summary(
    data, var_names=[v for v in data.posterior.data_vars if v != "pred"]
))

q = np.percentile(pred, [15.9, 50, 84.1], axis=0)

unc = np.mean([q[1]-q[0],q[2]-q[1]],axis=0)

fig=plt.figure(figsize=(24,6))
ax1=fig.add_subplot(131,projection='mollweide')
ax2=fig.add_subplot(132,projection='mollweide')
ax3=fig.add_subplot(133,projection='mollweide')

cbar_h=0.075
cbar_w=0.2
cbar_bottom=0.1

#ax.imshow(q[1].reshape(len(phi),len(theta)),extent=[0,360,-90,90],origin='lower',vmin=q[1].min(),vmax=q[1].max())
im=ax1.pcolor(
    phi-np.pi,
    theta,
    q[1].reshape((len(phi), len(theta))).T,
    vmin=17,vmax=19)


ax1.scatter(
    phi_obs-np.pi,
    theta_obs,
    c=y_obs,
    edgecolor="k",vmin=17,vmax=19)

cax=fig.add_axes([0.135, cbar_bottom, cbar_w, cbar_h])

fig.colorbar(im, orientation="horizontal",cax=cax,label='n(HI) (cm-3)')


pred_unc = np.mean([(q[2]-q[1]).reshape((len(phi), len(theta))).T, (q[1]-q[0]).reshape((len(phi), len(theta))).T],axis=0)
im=ax2.pcolor(
    phi-np.pi,
    theta,
    (unc/q[1]).reshape((len(phi), len(theta))).T / np.log(10) ,#pred_unc,
    cmap='gray')


ax2.plot(
    phi_obs-np.pi,
    theta_obs,
    'o',ms=7,mfc='none',mec='m')

cax2=fig.add_axes([0.41, cbar_bottom, cbar_w, cbar_h])
fig.colorbar(im, orientation="horizontal",cax=cax2,label='log N(HI) uncertainty (dex)')#label='n(HI) uncertainty (cm-3)')#


## I want this one to plot the fractional uncertainty!
phi_grid, theta_grid=np.meshgrid(phi,theta)
y_pred = griddata((phi_grid.flatten(),theta_grid.flatten()),q[1],(phi_obs,theta_obs))
residuals = (y_obs - y_pred)#/y_obs


eee=ax3.scatter(
    phi_obs-np.pi,
    theta_obs,
    c=residuals,
    edgecolor="k",cmap='PiYG',vmin=-0.1,vmax=0.1)

cax3=fig.add_axes([0.685, cbar_bottom, cbar_w, cbar_h])
fig.colorbar(eee, orientation="horizontal",cax=cax3,label='Residuals in n(HI) estimate')

"""
ndim=4
nbins=20
quantiles=[0.16,0.5,0.84]
truths=None
show_titles=False

fig, axes = plt.subplots(ndim, ndim, figsize=(12.5,9))
triangle.corner(data, bins=nbins, #labels=variable_names,
                      max_n_ticks=3,plot_contours=True,quantiles=quantiles,fig=fig,
                      show_titles=True,verbose=True,truths=truths,range=None)
"""

bins=50
alpha=0.3
c0='C0'
c1='C1'

colors=['C0','C1','C2','C3']

fig=plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
#ax5 = fig.add_subplot(235)

for i in range(len(data.posterior.log_amp.values)):
    ax1.hist(data.posterior.log_amp.values[i],bins=bins,alpha=alpha,color=colors[i])

    ax2.hist(data.posterior.log_scale.values[i],bins=bins,alpha=alpha,color=colors[i])


    ax4.hist(data.posterior.log_avg.values[i],bins=bins,alpha=alpha,color=colors[i])
    alpha += 0.1

#ax5.hist(data.posterior.log_c.values[0],bins=bins,alpha=alpha,color=c0)
#ax5.hist(data.posterior.log_c.values[1],bins=bins,alpha=alpha,color=c1)
#ax5.set_xlabel('log constant')

ax1.set_xlabel('log amp')
ax4.set_xlabel('log avg')
ax2.set_xlabel('log scale')


np.savetxt('NHI_column_map.txt',q[1])
np.savetxt('NHI_column_fitted_stars.txt',np.transpose(np.array([phi_obs,theta_obs,y_obs])))

