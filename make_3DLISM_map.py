import numpy as np
import matplotlib.pyplot as plt   
import pandas as pd
import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
from jax.config import config
import jaxopt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import spherical_to_cartesian

plt.ion()


config.update("jax_enable_x64", True)
pd.options.mode.chained_assignment = None



## Create fake data to model ### The column densities / HI distribution is set up later
nstars = 100
ls = np.random.uniform(0, 360, nstars) # Galactic l in degrees
bs = np.random.uniform(-90, 90, nstars) # Galactic b in degrees
ds = np.random.lognormal(2.3, 0.5, nstars)  # distance in pc
ds[ds > 40] = 39. # impose 40 pc maximum for simplicity
skycoords = SkyCoord(ls * u.degree,bs * u.degree,ds * u.pc, frame='galactic')
####################################################

#############################
# Initialize or read in the big grid where each cell knows its cartesian
# and spherical polar coordinates
# From Dustribution
#############################

#Copied from Dustribution - A spherical polar coords grid which also knows about its corresponding cartesian coords
def threeD_grid(l_lower, l_upper, n_l, b_lower, b_upper, n_b, d_min, d_max, n_d):  


    #Gives boundaries of l,b,d grid cells
    l_bounds = np.linspace(l_lower, l_upper, n_l+1) #+1: Need one more bound than the number of cells to get the last bound
    b_bounds = np.rad2deg(np.arcsin(np.linspace(np.sin(np.deg2rad(b_lower)), np.sin(np.deg2rad(b_upper)), n_b+1)))
    #d_bounds = np.logspace(np.log10(d_min), np.log10(d_max), n_d+1) #dist in pc and logspacing
    d_bounds = np.linspace(d_min, d_max, n_d+1) #dist in pc and linear spacing

    rows = []


    for i in range(len(l_bounds)-1): #-1 to remove the last bound element so that we don't do i+1 on last element since there is no +1
        for j in range(len(b_bounds)-1):
            for k in range(len(d_bounds)-1):

                if (i%10 == 0) & (j%10 == 0) & (k%10 == 0):
                    print(i,j,k)

                #Calc mid points of grid cells for the spherical polar coords
                l_mid = (l_bounds[i] + l_bounds[i+1])/2
                b_mid = (b_bounds[j] + b_bounds[j+1])/2
                #d_mid = 10**( ( np.log10(d_bounds[k]) + np.log10(d_bounds[k+1]) )/2 ) #log spacing distance
                d_mid = (d_bounds[k] + d_bounds[k+1])/2 #linear spaced distance
                

                #Calc mid points in Cart coords - required to place blob and fill density
                #Need to give the input as dist, lat, long (d,b,l)
                x_mid, y_mid, z_mid = spherical_to_cartesian(d_mid, np.deg2rad(b_mid), np.deg2rad(l_mid)) 


                rows.append([l_mid, b_mid, d_mid, x_mid, y_mid, z_mid]) 

    

    threeDGrid = pd.DataFrame(rows, columns=["pol_l", "pol_b", "pol_d", "cart_x", "cart_y", "cart_z"], dtype="float")


    return l_bounds, b_bounds, d_bounds, threeDGrid 





initialize_grid = False # will need to be set to True the first time you run it

if initialize_grid:

    l_lower=0
    l_upper=359.9
    n_l=100
    b_lower=-90
    b_upper=90
    n_b=100
    d_min=0
    d_max=40
    n_d=40


    l_bounds, b_bounds, d_bounds, threeDGrid = \
            threeD_grid(l_lower, l_upper, n_l, b_lower, b_upper, n_b, d_min, d_max, n_d)
    threeDGrid.to_pickle("threeDGrid.pkl")
    np.save("l_bounds.pkl",l_bounds, allow_pickle=True)
    np.save("b_bounds.pkl",b_bounds, allow_pickle=True)
    np.save("d_bounds.pkl",d_bounds, allow_pickle=True)

else:

    threeDGrid = pd.read_pickle("threeDGrid.pkl")
    l_bounds=np.load("l_bounds.pkl.npy", allow_pickle=True)
    b_bounds=np.load("b_bounds.pkl.npy", allow_pickle=True)
    d_bounds=np.load("d_bounds.pkl.npy", allow_pickle=True)
##################################


##################################
# Populate the grid with some clouds
###################################

def spheroid_calculation(x_array, y_array, z_array, cloud_dic): 
    # this calculates the distance of each grid point from the spheroid's center. The boundaries
    # of the spheroid are at 1
    # need to update this description, it's not 100% correct

    dist_from_spheroid_center = np.sqrt( ( (x_array - cloud_dic["x_center"]) / cloud_dic["x_scale"] )**2 + \
            ( (y_array - cloud_dic["y_center"]) / cloud_dic["y_scale"] )**2 + \
            ( (z_array - cloud_dic["z_center"]) / cloud_dic["z_scale"] )**2
                   )

    return dist_from_spheroid_center


# Let's have two uniform density spheroids units: pc] 
## Cloud 1 properties
cloud1 = {"x_scale":20, 
          "y_scale":18,
          "z_scale":70,
          "x_center":-8, 
          "y_center":8,
          "z_center":10,
          "density":0.04}

## Cloud 2 properties
cloud2 = {"x_scale":10, 
          "y_scale":50,
          "z_scale":20,
          "x_center":5,
          "y_center":-10,
          "z_center":-15,
          "density":0.08}

# Calculate the distance of each grid point from the spheroid center
dist1 = spheroid_calculation(threeDGrid['cart_x'],threeDGrid['cart_y'],threeDGrid['cart_z'],cloud1)
dist2 = spheroid_calculation(threeDGrid['cart_x'],threeDGrid['cart_y'],threeDGrid['cart_z'],cloud2)

threeDGrid['density'] = 0 # initialize the density grid
threeDGrid['density'][(dist1 < 1)] += cloud1["density"] # if grid point is inside cloud, add that cloud's density
threeDGrid['density'][(dist2 < 1)] += cloud2["density"] # same for cloud 2


# plot to see how it looks in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(threeDGrid['cart_x'][threeDGrid['density'] > 0], threeDGrid['cart_y'][threeDGrid['density'] > 0], threeDGrid['cart_z'][threeDGrid['density'] > 0], c=threeDGrid['density'][threeDGrid['density'] >0], marker='.', alpha=0.2)

################################
# Integrate the grid to get column densities
# From Dustribution
################################

# find where the stars are on the grid (from Dustribution)
l_ind = np.zeros_like(skycoords,dtype=int)
b_ind = l_ind.copy()
d_ind = l_ind.copy()
for i in range(len(skycoords)):

    l_ind[i] = np.argwhere(l_bounds < skycoords[i].l.value)[-1][0]
    b_ind[i] = np.argwhere(b_bounds < skycoords[i].b.value)[-1][0]
    d_ind[i] = np.argwhere(d_bounds < skycoords[i].distance.value)[-1][0]
#

## from Dustribution, modified
def integAllSource(source_dists, source_l, source_b, l_bounds, b_bounds, d_bounds, l_ind, b_ind, d_ind, density_samples):

    density_grid = density_samples.reshape(len(l_bounds)-1,len(b_bounds)-1,len(d_bounds)-1)
    integral_grid = jnp.cumsum(density_grid * (d_bounds[1:] - d_bounds[:-1]), axis=2) * 3.09e18 #3.09e18 cm/pc

    column_densities = jnp.zeros(len(skycoords))
    for i in range(len(column_densities)):


        column_densities=column_densities.at[i].set(integral_grid[l_ind[i],b_ind[i],d_ind[i]] + \
            ((integral_grid[l_ind[i],b_ind[i],d_ind[i]+1] - integral_grid[l_ind[i],b_ind[i],d_ind[i]]) * \
            (skycoords[i].distance.value - d_bounds[d_ind[i]])) )

    return column_densities

# calculate column densities of the sources
column_densities = integAllSource(jnp.array(skycoords.distance), jnp.array(skycoords.l), jnp.array(skycoords.b), l_bounds, b_bounds, d_bounds, l_ind, b_ind, d_ind, jnp.array(threeDGrid['density']))



#############################################
## Set up data to fit 
###########################################
y_truth = np.log10(column_densities) # log N(HI), units: dex
y_obs = y_truth + \
        np.random.normal(loc=0, scale=0.1, size=len(column_densities)) # add some random noise
yerr= 0.1 * np.ones_like(y_obs) # units: dex

X_obs_lbd = np.array([ls,bs,ds]).T
X_obs_cartesian = np.array([skycoords.cartesian.x, 
                        skycoords.cartesian.y, skycoords.cartesian.z]).T

X_grid = np.array([threeDGrid['cart_x'], threeDGrid['cart_y'], threeDGrid['cart_z']]).T
####################################################

## Plot the data ####################################
fig=plt.figure()
ax=fig.add_subplot(111,projection='mollweide')
c=ax.scatter(
    skycoords.l.radian-np.pi,
    skycoords.b.radian,
    c=y_obs,
    edgecolor="k",vmin=17.5,vmax=19.0)
ax.set_xlabel("l")
ax.set_ylabel("b")
ax.set_title("log10 N(HI) data")
plt.colorbar(c)

plt.figure()
plt.errorbar(ds,y_obs,yerr=yerr,fmt='ko')
plt.xlabel('Distance (pc)',fontsize=18)
plt.ylabel('N(HI)',fontsize=18)
#######################################################



## Define model to fit ################################
def numpyro_model(X_obs, yerr, y=None): 
    avg = numpyro.sample("log10_avg", dist.Cauchy(-1,0.5)) # log10 of the average number density (cm^{-3})
    amp = numpyro.sample("log_amp", dist.Cauchy(0,10)) # ln(amplitude) 
    length_scale1 = numpyro.sample("log_lengthscale1", dist.Normal(-1,0.5))
    #length_scale2 = numpyro.sample("log_lengthscale2", dist.Normal(-1,0.5))
    #length_scale3 = numpyro.sample("log_lengthscale3", dist.Normal(-1,0.5))

    log10_density = jnp.exp(amp) * kernels.ExpSquared(jnp.power(10,length_scale1))
                        #jnp.array([jnp.power(10,length_scale1),jnp.power(10,length_scale2),
                        #jnp.power(10,length_scale3)])  ) # model of the number density (cm^{-3})


    density = jnp.power(10,log10_density)

    column_densities = integAllSource(jnp.array(skycoords.distance), jnp.array(skycoords.l), jnp.array(skycoords.b), l_bounds, b_bounds, d_bounds, l_ind, b_ind, d_ind, density)
    
    gp=GaussianProcess(
        column_densities,
        X_obs,
        diag=yerr**2, 
        mean=jnp.exp(avg) )


    numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    if y is not None:
        numpyro.deterministic("pred", gp.predict(y, X_grid))
#######################################################




## set up NUTS ################################################
nuts_kernel = NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9)
mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        progress_bar=True,
        )
rng_key = jax.random.PRNGKey(34913)
################################################################

## Run the MCMC ################################################
mcmc.run(rng_key, X_obs_cartesian, yerr, y=y_obs)
###############################################################

## Get fit results ###########################################
samples = mcmc.get_samples()
pred = samples["pred"].block_until_ready()  # Blocking to get timing right
data = az.from_numpyro(mcmc)
print(az.summary(
data, var_names=[v for v in data.posterior.data_vars if v != "pred"]
         ))


