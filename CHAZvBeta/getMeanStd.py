#!/usr/bin/env python
import numpy as np
import calendar
import random
import time
import sys
import dask.array as da
import os
import gc
import pickle
import xarray as xr
import copy
import pandas as pd
import src.module_riskModel as mrisk
from netCDF4 import Dataset
from scipy import stats
from tools.util import argminDatetime
from tools.util import int2str,date_interpolation
from scipy.io import loadmat, netcdf_file
from datetime import datetime,timedelta
from tools.regression4 import calMultiRegression_coef_mean_water
from tools.regression4 import calMultiRegression_coef_mean_land

def getLonLatfromDistance(lonInit,latInit,dx,dy):
    er = 6371000 #km
    londis = 2*np.pi*er*np.cos(latInit/180*np.pi)/360.
    lon2 = lonInit+dx/londis
    latdis = 2*np.pi*er/360.
    lat2 = latInit+dy/latdis
    return lon2,lat2


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

def removeland(iS):
    b = np.int(iS.mean(keepdims=True))
    iT3 = -1
    if b<fstldmask.shape[1]:
        a = np.argwhere(fstldmask[:,b]==3)
        if a.size:
           if a.size>3:
              iT3 = a[0]+2
              fstlon[iT3:,b]=np.NaN
              fstlat[iT3:,b]=np.NaN
    return iT3

def func_first(x):
    if x.first_valid_index() is None:
        return None
    else:
        return x.first_valid_index()
def func_last(x):
    if x.last_valid_index() is None:
        return None
    else:
        return x.last_valid_index()



def get_predictors(iiS,block_id=None):
	iS = np.int(iiS.mean(keepdims=True))
       	predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
	if bt['StormYear'][iS]>1980:
		fileName = ipath+str(bt['StormYear'][iS])+'_'+ens+'.nc'
		nc = Dataset(fileName,'r',format='NETCDF3_CLASSIC')
        	xlong = nc.variables['Longitude'][:]
	        xlat = nc.variables['Latitude'][:]
        	PIVmax = nc.variables['PI2'][:,:,:]*1.94384449 #m/s - kt
                u250 = nc.variables['ua2502'][:]
                u850 = nc.variables['ua8502'][:]
                v250 = nc.variables['va2502'][:]
                v850 = nc.variables['va8502'][:]
		u = u250-u850
		v = v250-v850
		meanrh = nc.variables['hur2'][:]
	        xxlong,xxlat = np.meshgrid(xlong,xlat)
        	del xlong, xlat
        	nc.close()
		for it in range(0,bt['StormLon'][:,iS].shape[0],1):	
			if bt['Time'][it,iS] > 0:
			   distance = np.empty(xxlong.shape,dtype=float)
			   er = 6371.0 #km
			   londis = 2*np.pi*er*np.cos(xxlat/180*np.pi)/360
			   dx = londis*(xxlong - bt['StormLon'][it,iS])
			   dy = 110 * (xxlat - bt['StormLat'][it,iS])
			   distance = np.sqrt(dx*dx+dy*dy)
			   (j0,i0) = np.unravel_index(np.argmin(distance),distance.shape)

			   var,radius1,radius2 =  date_interpolation(datetime(1800,01,01)+timedelta(days = bt['Time'][it,iS]),PIVmax),0,500
			   bt['PIwspdMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
			   bt['PIwspd'][it,iS] = var[j0,i0]

			   var,radius1,radius2 =  date_interpolation(datetime(1800,01,01)+timedelta(days = bt['Time'][it,iS]),u),200,800
			   bt['UShearMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
			   bt['UShear'][it,iS] = var[j0,i0]

			   var,radius1,radius2 =  date_interpolation(datetime(1800,01,01)+timedelta(days = bt['Time'][it,iS]),v),200,800
			   bt['VShearMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
			   bt['VShear'][it,iS] = var[j0,i0]

			   var,radius1,radius2 =  date_interpolation(datetime(1800,01,01)+timedelta(days = bt['Time'][it,iS]),meanrh),200,800
			   bt['rhMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
			   bt['rh'][it,iS] = var[j0,i0]

			   var,radius1,radius2 = copy.copy(ldmask),0,300
			   var[var==0] = -1.
			   var[var==3] = 0.0
			   bt['landmaskMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
			   bt['landmask'][it,iS] = var[j0,i0] 

	return iS

def getStormTranslation(lon,lat,time):

        er = 6371.0 #km
	delt = (time[1:]-time[:-1])*24*60*60
	dlon = lon[1:]-lon[:-1]
	dlat = lat[1:]-lat[:-1]
        speed = np.zeros(lon.shape[0],dtype=float)+float('nan')
        sdir = np.zeros(lon.shape[0],dtype=float)+float('nan')
        londis = 2*np.pi*er*np.cos(lat[0:-1]/180*np.pi)/360
        dx = londis*(dlon)
        dy = 110*(dlat)
        distance = np.sqrt(dx*dx+dy*dy) #km
        sdir[:-1]=np.arctan2(dlat,dlon)
	speed[:-1]=distance*1000./delt #m/s

        return sdir,speed

			
def getSpeedDir(iiS,block_id=None):
    iS = np.int(iiS.mean(keepdims=True))
    if (bt['StormLat'][:,iS]==bt['StormLat'][:,iS]).any():
        it1 = np.argwhere(bt['StormLat'][:,iS]==bt['StormLat'][:,iS])[0,0]
        it2 = np.argwhere(bt['StormLat'][:,iS]==bt['StormLat'][:,iS])[-1,0]
        if it2 - it1 >=2:
           bt['trDir'][it1:it2,iS],bt['trSpeed'][it1:it2,iS]=\
             getStormTranslation(bt['StormLon'][it1:it2,iS],\
             bt['StormLat'][it1:it2,iS],bt['Time'][it1:it2,iS])
    return iS

class mybt:
	def __init__(self,bt_array):
		for iv in bt_array.keys():
			if 'Time' not in iv:
				setattr(self, iv, bt_array[iv])
			else:
				start = pd.Timestamp('1800-01-01').to_pydatetime()
				setattr(self, iv, start+np.int_(bt_array[iv]*24)*timedelta(hours=1))

def calCoefficient_water_guess(ty1, ty2, ih):
    """
    calculate first guess coefficient from OLS
    """
    from tools.regression4 import calMultiRegression_coef_mean_water
    from tools.regression4 import calMultiRegression_coef_mean_land

    ### calculatae coefficients
    predictors = ['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2']
    result_w,meanX_w,meanY_w,stdX_w,stdY_w = \
       calMultiRegression_coef_mean_water(bt,ih,\
       ty1,ty2,predictors,'dVmax')
    predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
    result_l,meanX_l,meanY_l,stdX_l,stdY_l = \
       calMultiRegression_coef_mean_land(bt,ih,\
       ty1,ty2,predictors,'dVmax')

    return(result_w,meanX_w,meanY_w,stdX_w,stdY_w,result_l,meanX_l,meanY_l,stdX_l,stdY_l)


ens = 'r1i1p1f1'
ipath = './pre/'
iy1,iy2 = 1981,2012
global ldmask
landmaskfile = 'input/landmask.nc'
llon, llat,dummy = get_landmask(landmaskfile)
ldmask = dummy[-12::-24,::24]

if True:
	global bt
	bt = mrisk.resize_basinData()

	bt['StormYear'] = np.int16(bt['StormYear'])
	time1 = time.time()
	nS = bt['div200'].shape[1]
	diS = da.from_array(np.arange(0,nS,1).astype(dtype=np.int32),chunks=(1,))
	niS = np.arange(0,nS,1).astype(dtype=np.int32)
	new = da.map_blocks(getSpeedDir,diS,chunks=(1,),dtype=diS.dtype)
	new.compute()
	print 'down translation speed', time.time()-time1
	del new
	gc.collect()

	### ca#lculate tracks ###
	#time1 =time.time()
	for iS in niS: 
	    #print iS 
	    get_predictors(np.array([iS,iS]))
	print 'serial done', time.time()-time1

	#time1 = time.time()
	#new = da.map_blocks(get_predictors,diS,chunks=(1,1))
	#a = new.compute()
	print ' done calPredictors', time.time()-time1
	gc.collect()
	bt1 = copy.deepcopy(bt)
	del bt
	bt = mybt(bt1)
	with open(ipath+'trackPredictorsbt_obs.pik','w+')as f:
	     pickle.dump(bt,f)
	f.close()
else:
	with open(ipath+'trackPredictorsbt_obs.pik','r')as f:
	     bt = pickle.load(f)
	f.close()

result_w,meanX_w,meanY_w,stdX_w,stdY_w,result_l,meanX_l,meanY_l,stdX_l,stdY_l = calCoefficient_water_guess(iy1,iy2+1,12)

meanX_w = np.array(meanX_w)
stdX_w = np.array(stdX_w)
meanY_w = np.array(meanY_w)
stdY_w = np.array(stdY_w)
meanX_l = np.array(meanX_l)
stdX_l = np.array(stdX_l)
meanY_l = np.array(meanY_l)
stdY_l = np.array(stdY_l)

d2 = np.empty(meanX_w.shape[0])
d3 = np.empty(stdX_w.shape[0])
d7 = np.empty(meanX_l.shape[0])
d8 = np.empty(stdX_l.shape[0])


ds = xr.Dataset({
  'meanX_w': xr.DataArray(
                 data = meanX_w,
                 dims = ['d2']),


  'stdX_w': xr.DataArray(
                 data = stdX_w,
                 dims = ['d3']),


  'meanY_w': xr.DataArray(
                 data = meanY_w),


  'stdY_w': xr.DataArray(
                 data = stdY_w),

  'meanX_l': xr.DataArray(
                 data = meanX_l,
                dims = ['d7']),


  'stdX_l': xr.DataArray(
                 data = stdX_l,
                dims = ['d8']),


  'meanY_l': xr.DataArray(
                 data = meanY_l),


  'stdY_l': xr.DataArray(
                 data = stdY_l),

        }
    )


filename = ipath+'coefficient_meanstd.nc'
ds.to_netcdf(filename)
