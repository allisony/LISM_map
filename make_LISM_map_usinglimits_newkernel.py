import numpy as np
import matplotlib.pyplot as plt   
from matplotlib import rc, rcParams
import pandas as pd
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
from numpyro.contrib.control_flow import scan
import arviz as az
import corner as triangle
from scipy.interpolate import griddata

config.update("jax_enable_x64", True)
pd.options.mode.chained_assignment = None


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
####################################################


## Define Haversine distance class for use with tinygp ##################
class GreatCircleDistance(kernels.stationary.Distance):
    def distance(self, X1, X2):
        if jnp.shape(X1) != (3,) or jnp.shape(X2) != (3,):
            raise ValueError(
                "The great-circle distance is only defined for unit 3-vector"
            )
        return jnp.arctan2(jnp.linalg.norm(jnp.cross(X1, X2)), (X1.T @ X2))
##############################################################################


## Read in the data -- see format_spreadsheet.py ###
df = pd.read_csv('targets/NHI_data_July2024.csv')
####################################################

## Make a spherical grid#############################
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
####################################################

## Put star coordinates on that grid ###############
skycoords = SkyCoord(ra= df['RA'] * u.degree, dec = df['DEC'] * u.degree)
X_obs_all = np.array(skycoords.cartesian.xyz.T) # shape (100,3) -- for unit vectors
theta_obs_all = df['DEC'].values * np.pi/180. 
phi_obs_all  = df['RA'].values * np.pi/180.
####################################################


## Set up data to fit ##############################
y_obs_all = df['N(HI)'].values
yerr_all= df['N(HI) uncertainty'].values

d_all = df['distance (pc)']

run_loop = False
n_runs = 100

if True:

    mask =  (d_all<=10) #& (d_all <= 100) # # d_all<=10 #
    filename_precursor = 'all_inside_10pc'

    y_obs = y_obs_all[mask]
    yerr = yerr_all[mask]
    theta_obs = theta_obs_all[mask]
    phi_obs = phi_obs_all[mask]
    X_obs = X_obs_all[mask]
    d = d_all[mask]

    lower_mask = d_all< 0 #10  ## EDIT
    X_obs_lower = X_obs_all[lower_mask]
    y_lower = y_obs_all[lower_mask]
    yerr_lower = yerr_all[lower_mask]
    
    upper_mask = d_all > 10 ### EDIT
    X_obs_upper = X_obs_all[upper_mask]
    y_upper = y_obs_all[upper_mask]
    yerr_upper = yerr_all[upper_mask]


####################################################

## Plot the data ####################################
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
#######################################################

#import pdb; pdb.set_trace()


## Define model to fit ################################
def numpyro_model(X_obs, yerr, y=None, X_obs_upper=None, y_upper=None, yerr_upper = None, X_obs_lower=None, y_lower=None, yerr_lower=None): 
    avg1 = numpyro.sample("log_avg", dist.Cauchy(18,2)) # do I need to change this?? it should be log?
    amp1 = numpyro.sample("log_amp", dist.Cauchy(0,10))
    scale1 = numpyro.sample("log_scale", dist.Normal(-1,0.5))

    amp2 = numpyro.sample("log_amp2", dist.Cauchy(0,10))
    scale2 = numpyro.sample("log_scale2", dist.Normal(-2,0.5))

    kernel = jnp.exp(amp1) * kernels.Exp(jnp.exp(scale1), distance=GreatCircleDistance()) + jnp.exp(amp2) * kernels.Exp(jnp.exp(scale2), distance=GreatCircleDistance())

    gp=GaussianProcess(
        kernel,
        X_obs,
        diag=yerr**2, #+ (sigma1 * jnp.log(10) * y)**2, 
        mean=jnp.exp(avg1))


    # For uncensored data
    if y is not None:
        numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    # For upper limits, incorporating uncertainties
    if X_obs_upper is not None and y_upper is not None and yerr_upper is not None:
        # Predict at X_obs_upper locations
        mu_upper, var_upper = gp.predict(y, X_obs_upper, return_var=True)
        # Adjust the variance to include the uncertainty in the upper limit
        #adjusted_var_upper = var_upper + yerr_upper**2

        #upper_limit_likelihood = 1 - dist.Normal(mu_upper, jnp.sqrt(adjusted_var_upper)).cdf(y_upper)
        #numpyro.factor("upper_limit", jnp.log(upper_limit_likelihood))
        #upper_distance = (mu_upper - jnp.sqrt(var_upper)) < (y_upper + yerr_upper)
        upper_distance = (y_upper + yerr_upper) - (mu_upper - jnp.sqrt(var_upper))
        upper_limit_likelihood = jax.nn.sigmoid(1e3*upper_distance)
        #print("upper")
        #print(upper_distance, upper_limit_likelihood, jnp.log(upper_limit_likelihood))
        numpyro.factor("upper_limit", jnp.log(upper_limit_likelihood))
        

    # For lower limits, incorporating uncertainties
    if X_obs_lower is not None and y_lower is not None and yerr_lower is not None:
        # Predict at X_obs_lower locations
        mu_lower, var_lower = gp.predict(y, X_obs_lower, return_var=True)
        # Adjust the variance to include the uncertainty in the lower limit
        #adjusted_var_lower = var_lower + yerr_lower**2
        #lower_limit_likelihood = dist.Normal(mu_lower, jnp.sqrt(adjusted_var_lower)).cdf(y_lower)
        #numpyro.factor("lower_limit", jnp.log(lower_limit_likelihood))
        #lower_distance = (mu_lower + jnp.sqrt(var_lower)) > (y_lower - yerr_lower)
        lower_distance = (mu_lower + jnp.sqrt(var_lower)) - (y_lower - yerr_lower)
        lower_limit_likelihood = jax.nn.sigmoid(1e3*lower_distance)
        #print("lower")
        #print(lower_distance, lower_limit_likelihood, jnp.log(lower_limit_likelihood))

        numpyro.factor("lower_limit", jnp.log(lower_limit_likelihood))

        
    if y is not None:
        numpyro.deterministic("pred", gp.predict(y, X_grid))
#######################################################


if run_loop:

    q_array = np.zeros((n_runs,int(len(phi)*len(theta))))

    log_amp_array = np.zeros(n_runs)
    log_avg_array = log_amp_array.copy()
    log_scale_array = log_amp_array.copy()

    log_amp2_array = log_amp_array.copy()
    log_scale2_array = log_amp_array.copy()
    

    for i in range(n_runs):

        print(i, n_runs)

        y_obs_samp = np.random.normal(loc=y_obs, scale=yerr)

        ## set up NUTS ################################################
        nuts_kernel = NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9)
        mcmc = MCMC(
                    nuts_kernel,
                    num_warmup=1000,
                    num_samples=2000,
                    num_chains=2,
                    progress_bar=True,
                    )
        rng_key = jax.random.PRNGKey(34913)

        mcmc.run(rng_key, X_obs, yerr, y=y_obs_samp)
        samples = mcmc.get_samples()
        pred = samples["pred"].block_until_ready()  # Blocking to get timing right
        data = az.from_numpyro(mcmc)
        q_array[i,:] = np.median(pred,axis=0)
        log_amp_array[i] = np.median(data.posterior.log_amp.values.reshape(4000)) # should automate this = 2 * 2000 (num_samples * num_chains)
        log_avg_array[i] = np.median(data.posterior.log_avg.values.reshape(4000))
        log_scale_array[i] = np.median(data.posterior.log_scale.values.reshape(4000))
        log_amp2_array[i] = np.median(data.posterior.log_amp2.values.reshape(4000)) # should automate this = 2 * 2000 (num_samples * num_chains)
        log_scale2_array[i] = np.median(data.posterior.log_scale2.values.reshape(4000))

    q = np.percentile(q_array, [15.9, 50, 84.1], axis=0)

else:


    ## set up NUTS ################################################
    nuts_kernel = NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=1000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
        )
    rng_key = jax.random.PRNGKey(34913)
    ################################################################

    ## Run the MCMC ################################################
    mcmc.run(rng_key, X_obs, yerr, y=y_obs, X_obs_upper=X_obs_upper, y_upper=y_upper, yerr_upper = yerr_upper, X_obs_lower=X_obs_lower, y_lower=y_lower, yerr_lower=yerr_lower)
    ###############################################################

    ## Get fit results ###########################################
    samples = mcmc.get_samples()
    pred = samples["pred"].block_until_ready()  # Blocking to get timing right
    data = az.from_numpyro(mcmc)
    print(az.summary(
        data, var_names=[v for v in data.posterior.data_vars if v != "pred"]
         ))

    q = np.percentile(pred, [15.9, 50, 84.1], axis=0)
    

unc = np.mean([q[1]-q[0],q[2]-q[1]],axis=0)
#######################################################

## Plot fit results ###################################
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
        edgecolor="k",cmap='PiYG',vmin=-0.3,vmax=0.3)

cax3=fig.add_axes([0.685, cbar_bottom, cbar_w, cbar_h])
fig.colorbar(eee, orientation="horizontal",cax=cax3,label='Residuals in N(HI) estimate')

#####################################################################
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
## Plot histogram of results #########################################
bins=50
alpha=0.3
c0='C0'
c1='C1'

colors=['C0','C1','C2','C3']

fig=plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(151)
ax2 = fig.add_subplot(152)
ax4 = fig.add_subplot(153)
ax5 = fig.add_subplot(154)
ax6 = fig.add_subplot(155)


if run_loop:
    ax1.hist(log_amp_array,bins=bins,alpha=alpha)
    ax2.hist(log_avg_array,bins=bins,alpha=alpha)
    ax4.hist(log_scale_array,bins=bins,alpha=alpha)
    ax5.hist(log_amp2_array,bins=bins,alpha=alpha)
    ax6.hist(log_scale2_array,bins=bins,alpha=alpha)


else:

    for i in range(len(data.posterior.log_amp.values)):
        ax1.hist(data.posterior.log_amp.values[i],bins=bins,alpha=alpha,color=colors[i])

        ax2.hist(data.posterior.log_scale.values[i],bins=bins,alpha=alpha,color=colors[i])


        ax4.hist(data.posterior.log_avg.values[i],bins=bins,alpha=alpha,color=colors[i])
        ax5.hist(data.posterior.log_amp2.values[i],bins=bins,alpha=alpha,color = colors[i])
        ax6.hist(data.posterior.log_scale2.values[i],bins=bins,alpha=alpha,color=colors[i])

        alpha += 0.1

#ax5.hist(data.posterior.log_c.values[0],bins=bins,alpha=alpha,color=c0)
#ax5.hist(data.posterior.log_c.values[1],bins=bins,alpha=alpha,color=c1)
#ax5.set_xlabel('log constant')

ax1.set_xlabel('log amp')
ax4.set_xlabel('log avg')
ax2.set_xlabel('log scale')
ax5.set_xlabel('log amp2')
ax6.set_xlabel('log scale2')
    #######################################################################


## Save fit results to file ############################################
if True:
    func_dict = {
    "15.9%": lambda x: np.percentile(x, 15.9),
    "median": lambda x: np.percentile(x, 50),
    "84.1%": lambda x: np.percentile(x, 84.1),
    }
    table = az.summary(
        data, var_names=[v for v in data.posterior.data_vars if v != "pred"], stat_funcs=func_dict
         )
    np.savetxt('morecomplicatedkernel/NHI_column_map_'+filename_precursor +'_upperlowerlimits.txt',np.transpose(q))
    np.savetxt('morecomplicatedkernel/NHI_column_fitted_stars_' + filename_precursor + '_upperlowerlimits.txt',np.transpose(np.array([phi_obs,theta_obs,y_obs])))
    table.to_csv('morecomplicatedkernel/Bestfit_hyperparameters_' + filename_precursor + '_upperlowerlimits.csv')
#######################################################################
