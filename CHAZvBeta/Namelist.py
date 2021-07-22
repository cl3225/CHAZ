#!/usr/bin/env python
from scipy.io import loadmat, netcdf_file

def get_landmask(filename):
    """
    read 0.25degree landmask.nc 
    output:
    lon: 1D
    lat: 1D
    landmask:2D
  
    """
    f = netcdf_file(filename)
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    landmask = f.variables['landmask'][:,:]
    f.close()

    return lon, lat, landmask

### Experiment settings
Model = ''
ENS = ''
TCGIinput = ''
CHAZ_ENS_0 = 
CHAZ_ENS = 
CHAZ_Int_ENS = 
PImodelname = '' ### No longer need
### CHAZ parameters
monthlycsv = ''
dailycsv = ''
uBeta = -2.5
vBeta = 1.0
survivalrate = 0.78
seedN = 1 #annual seeding rate for random seeding, not yet ready to use
landmaskfile = 'input/landmask.nc'
ipath = 'input/'
opath = 'input/bt_global_predictors.nc'
pre_path = 'pre/'
output_path = 'output/'
Year1 =  
Year2 = 
llon, llat,lldmask = get_landmask(landmaskfile)
ldmask = lldmask[-12::-24,::24]
lldmask = lldmask[::-1,:]
#####################################################
# Preprocesses                                   ####
# ignore variavbles when run Preprocess is False ####
#####################################################
runPreprocess = False 
calWind = False 
calpreProcess = False 
calA = False
###################################################
# CHAZ                                         ####
# ignore variavbles when run CHAZ is False     ####
###################################################
runCHAZ=False
### genesis 
calGen = False 
### track
calBam = False
### intensiy
calInt = False

