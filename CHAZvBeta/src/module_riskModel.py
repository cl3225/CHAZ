#!/usr/bin/env python
import numpy as np
import calendar
import random
import time
import sys
import dask.array as da
import os
import gc
import xarray as xr
import copy
import pandas as pd
import Namelist as gv
import netCDF4 as nc
from netCDF4 import Dataset
from scipy import stats
from tools.util import int2str,date_interpolation
from scipy.io import loadmat, netcdf_file
from datetime import datetime,timedelta
from tools.regression4 import calMultiRegression_coef_mean_water
from tools.regression4 import calMultiRegression_coef_mean_land


def resize_basinData():
        """
        read in global data and make the new bt with same length
	this step can be elimated if we are using ibtracks in the future CHAZ development	
        """
        basinName = ['atl','wnp','enp','ni','sh']
	nd = 0
        for iib in range(0,len(basinName),1):
         ib = basinName[iib]
         f =gv.ipath + 'bt_'+ib+'.nc'          
	 #bt1 = nc.Dataset(f)
	 bt1 = xr.open_dataset(f)
         if iib == 0:
	    maxCol = bt1['PIslp'].shape[0]
	 else:
	    maxCol = np.nanmax([maxCol,bt1['PIslp'].shape[0]])
	 ## in bt1, the time is datenumber start from 1800,01,01,0,0. So if datenumber is 0 means there is no data
	 nd += bt1['PIslp'].shape[1]
	bt = {}
	for iib in range(0,len(basinName),1):
	 	bt1 = xr.open_dataset(f)
	        for iv in bt1.variables.keys():
			if iib == 0:
				if np.size(bt1.variables[iv].shape) >1:
				   bt[iv] = np.zeros([maxCol,bt1.variables[iv].shape[1]])*np.float('nan')
				   bt[iv][:bt1.variables[iv].shape[0],:] = bt1.variables[iv].values
				else:
				   bt[iv] = bt1.variables[iv].values
			else:
				if np.size(bt1.variables[iv].shape) >1:
				   dummy = np.zeros([maxCol,bt1.variables[iv].shape[1]])*np.float('nan')
				   dummy[:bt1.variables[iv].shape[0],:] = bt1.variables[iv].values
				   bt[iv] = np.hstack([bt[iv],dummy])
				   del dummy
				else:
				   bt[iv] = np.hstack([bt[iv],bt1.variables[iv].values])

	  	
         	del bt1
	for iv in bt.keys():
		if ((np.size(bt[iv].shape) >1) and ('Time' not in iv)):
			bt[iv][bt['Time']==0] = np.float('nan')
	bt['Time'][bt['Time']!=bt['Time']]=0
        return bt

def TCgiSeeding (gxlon,gxlat,gi,climInitLon,climInitLat,climInitDate,ratio):
    '''
        This is a subroutine for calculate genesis location based on gi.
        gi (Month X nlat X nlon): 
                gi is tropical cyclone genesis index. We use nansum(gi) to for 
                total number of seeds, and then randomly select location using gi 
                as genesis probability.
        gxlon (nlat X nlon): mesh-grid longitude
        gxlat (nlat X nlon): mesh-grid latitude 
        climInitLon,climInitLat,climInitDate: list contains previous seeds information
        ratio: how much more seeds we should gives, based on the survival rate
    '''
    # im - iterating through months
    for im in range(0,12,1):
        xk,pk = np.arange(gi[im,:,:].size),gi[im,:,:].ravel()
        dummy = xk*pk
        xk,pk = xk[dummy==dummy],pk[dummy==dummy]
        custm = stats.rv_discrete(name='custm', values=(xk, pk/pk.sum()))
        r = custm.rvs(size=np.round(pk.sum())*ratio)
        iix = gxlon.ravel()[r.ravel()]
        iiy = gxlat.ravel()[r.ravel()]
        iday = np.random.choice(np.arange(calendar.monthrange(iy,im+1)[1]),size=np.round(pk.sum()),replace=True)
        if iday.size>0:
           #iterating through days to find initial longitude, latitude, date
           for id in range(iday.size):
		   x1 = np.arange(iix[id]-1,iix[id]+1.01,0.01)
		   y1 = np.arange(iiy[id]-1,iiy[id]+1.01,0.01)
		   xx = np.random.choice(x1,1)
		   yy = np.random.choice(y1,1)
		   climInitDate.append(datetime(iy,im+1,iday[id]+1,0,0))
		   climInitLon.append(xx)
		   climInitLat.append(yy)
    return climInitLon,climInitLat,climInitDate

def calF(nday):
     '''
     This function finds the fourier function.
     nday: number of days
     '''
     #### fourier function
     dt = 1.0*60*60 #1 hr 
     #T = np.float(nday) 
     T = np.float(15)
     N = 15
     nt = np.arange(0,nday*60*60*24,dt)
     F = np.zeros([nt.shape[0],4])
     F1 = np.zeros([nt.shape[0],4])

     #F = np.zeros([24,4])
     #F1 = np.zeros([24,4])
     for iff in range(0,4,1):
         X = np.zeros([15])
         X = [random.uniform(0,1) for iN in range(N)]
         for itt in range(nt.shape[0]):
         #for itt in range(24):
             F[itt,iff] = (np.sqrt(2.0/np.sum([iN**(-3.0) for iN in range(1,N+1,1)]))*\
                       np.sum([(iN**(-3.0/2.0))*np.sin(2.0*np.pi*(iN*itt/(24.*T)+X[iN-1]))
                       for iN in range(1,N+1,1)]))

     return F


def getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate,uBeta,vBeta):
       '''
       This function predicts the track.
       u250: zonal wind time series at 250 hPA.
       v250: meridional wind time series at 250 hPA.
       u850: zonal wind time series at 850 hPA.
       v850: meridional wind time series at 850 hPA.
       dt: time differential.
       fstLon: longitude 
       fstLat: latitude
       fstDate: date
       '''
       #### modify Beta
       earth_rate = 7.2921150e-5 #mean earth rotation rate in radius per second
       r0 = 6371000 # mean earth radius in m
       lat0 = np.arange(-90,100,10)
       phi0 = lat0/180.*np.pi #original latitude in radian (ex. at 15 degree N)
       beta0 = 2.0*earth_rate*np.cos(phi0)/r0 # per second per m
       beta0 = beta0/beta0[10]
       ratio = np.interp(fstLat,np.arange(-90,100,10),beta0)
       uBeta = uBeta*ratio
       vBeta = vBeta*ratio
       ################

       alpha = 0.8
       uTrack = alpha*u850+(1.-alpha)*u250+uBeta
       vTrack = alpha*v850+(1.-alpha)*v250+vBeta*np.sign(np.sin(fstLat*np.pi/180.))
       dx = uTrack*dt
       dy = vTrack*dt
       lon2,lat2 = getLonLatfromDistance(fstLon,fstLat,dx,dy)
       fstLon,fstLat = lon2,lat2
       fstDate += timedelta(seconds=dt)
       #print uBeta, vBeta,fstLon,fstLat
       return fstLon,fstLat,fstDate

def getLonLatfromDistance(lonInit,latInit,dx,dy):
    '''
    This function calculates the latitude from the distance.
    lonInit: initial longitude
    latInit: initial latitude
    dx: x differential
    dy: y differential
    '''
    # calculate latitude and longitude from distance
    er = 6371000 #km
    # calculate longitude distance from initial longitude
    londis = 2*np.pi*er*np.cos(latInit/180*np.pi)/360.
    # calculate longitude
    lon2 = lonInit+dx/londis
    # calculate latitude distance 
    latdis = 2*np.pi*er/360.
    # calculate latitude
    lat2 = latInit+dy/latdis
    return lon2,lat2


def bam(iS,block_id=None):
     '''
     This function uses a beta advection model to find track. 
     '''
     # beta-advection model
     dt = 1.0*60*60 # 1 hour
     T = 15
     N = 15
     F = calF(15)
     nt = np.arange(0,T*60*60*24,dt)
     b = np.int(iS.mean(keepdims=True))
     #b = iS
     # initializing date, longitude, latitude
     fstDate = climInitDate[b]
     fstLon = climInitLon[b]
     fstLat = climInitLat[b]
     fstlon[0,b] = fstLon
     fstlat[0,b] = fstLat
     fstldmask[0,b] = 0
     # calculating end hours, end date
     endhours = fstlon.shape[0]-1
     endDate = climInitDate[b] + timedelta(hours = endhours)
     count,year0,month0,day0 = 1,0,0,0

     while fstDate < endDate:
	if (fstDate.year != year0) :
	   fileName = gv.ipath+str(fstDate.year)+'_'+ens+'.nc'
           if os.path.isfile(fileName):
		nc = Dataset(fileName,'r',format='NETCDF3_CLASSIC')
		u250m = nc.variables['ua2502'][:]
		u850m = nc.variables['ua8502'][:]
		v250m = nc.variables['va2502'][:]
		v850m = nc.variables['va8502'][:]
		xlong = nc.variables['Longitude'][:]
		xlat = nc.variables['Latitude'][:]
		xxlong,xxlat = np.meshgrid(xlong,xlat)

	        if fstDate.day != day0:
			u250m2d = date_interpolation(fstDate,u250m)
			v250m2d = date_interpolation(fstDate,v250m)
			u850m2d = date_interpolation(fstDate,u850m)
			v850m2d = date_interpolation(fstDate,v850m)

			FileAName =\
		 	gv.ipath+'A_'+str(fstDate.year)+int2str(fstDate.month,2)+'.nc'
			ncA = Dataset(FileAName,'r',format='NETCDF3_CLASSIC')
			A = ncA.variables['A'][:,fstDate.day-1,:,:]
			ncA.close()
			day0 = fstDate.day

		distance = np.sqrt((fstLon-xxlong)**2+(fstLat-xxlat)**2)
		iy,ix = np.unravel_index(np.argmin(distance),distance.shape)
		iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])
		iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])

		iit = np.mod(count,nt.shape[0])
		u250 = u250m2d[iy1:iy2+1,ix1:ix2+1]+A[0,iy1:iy2+1,ix1:ix2+1]*F[iit,0]
		v250 = v250m2d[iy1:iy2+1,ix1:ix2+1]+A[1,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+A[2,iy1:iy2+1,ix1:ix2+1]*F[iit,1]
		u850 = u850m2d[iy1:iy2+1,ix1:ix2+1]+A[3,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
					     A[4,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[5,iy1:iy2+1,ix1:ix2+1]*F[iit,2]
		v850 = v850m2d[iy1:iy2+1,ix1:ix2+1]+A[6,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
					     A[7,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[8,iy1:iy2+1,ix1:ix2+1]*F[iit,2]+\
					     A[9,iy1:iy2+1,ix1:ix2+1]*F[iit,3]

		u250 = np.nanmean(u250)
		u850 = np.nanmean(u850)
		v250 = np.nanmean(v250)
		v850 = np.nanmean(v850)

		fstLon, fstLat, fstDate = getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate,uBeta,vBeta)
		if ((fstLon<0.0) or (fstLon>360) or (fstLat<-60) or (fstLat>60)):
              		#print b, 'break for going to the space'
              		break
           	fstlon[count,b] = fstLon
           	fstlat[count,b] = fstLat
		fstldmask[count,b] = np.rint(np.nanmean(ldmask[iy1:iy2+1,ix1:ix2+1])) 
           	count += 1
	   else:
		print 'no'+fileName
		break
     return b

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


class fst2bt(object):
    """
      convert data format from fst to be format of bt object.
    """
    def __init__(self,data):
        self.StormId = np.arange(0,data['lon'].shape[1],1)
        self.StormYear = []
        self.StormInitMonth = []
        for iS in self.StormId:
            if data['Time'][0,iS] is not None:
               self.StormYear.append(data['Time'][0,iS].year)
               self.StormInitMonth.append(data['Time'][0,iS].month)
            else:
               self.StormYear.append(1800)
               self.StormInitMonth.append(1)
        self.StormYear = np.array(self.StormYear)
        self.StormInitMonth = np.array(self.StormInitMonth)
        for iS in range(data['Time'].shape[1]):
            data['Time'][:,iS] = np.array([datetime(1800,1,1,0) if v is None else v for v in data['Time'][:,iS]])
        data['Time'][data['lon']!=data['lon']] = datetime(1800,1,1,0)
        data['lon'][data['Time']==datetime(1800,1,1,0)] = np.float('Nan')
        self.StormLon = data['lon']
        self.StormLat = data['lat']
        self.Time = data['Time']
        self.PIwspd = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.PIslp = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.PIwspdMean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.dPIwspd = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.PIslpMean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.UShearMean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.UShear = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.VShearMean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.VShear = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.div200Mean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.div200 = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.T200Mean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.T200 = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.rh500_300Mean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.rh500_300 = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.rhMean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.rh = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.T100Mean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.T100 = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.dThetaEMean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.dThetaE = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.dThetaEs = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.dThetaEsMean = np.empty(data['lon'].shape,dtype=float)*np.float('Nan')
        self.landmask = data['ldmask']
        self.landmaskMean = data['ldmask']
        self.trSpeed = np.zeros(data['lon'].shape)*np.float('Nan')
        self.trDir = np.zeros(data['lon'].shape)*np.float('Nan')
        self.dVdt = np.zeros(data['lon'].shape)

def func_first(x):
    '''
    This function returns the first non NA/Null value.
    '''
    if x.first_valid_index() is None:
        return None
    else:
        return x.first_valid_index()
def func_last(x):
    '''
    This function returns the last non NA/Null value.
    '''
    if x.last_valid_index() is None:
        return None
    else:
        return x.last_valid_index()



def get_predictors(iiS,block_id=None):
        '''
        This function returns an object containing the predictors.
        '''
	iS = np.int(iiS.mean(keepdims=True))
       	predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
	if bt.StormYear[iS]>=1980:
		fileName = gv.ipath+str(bt.StormYear[iS])+'_'+ens+'.nc'
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
		meanrh1 = nc.variables['hur2'][:]
	        xxlong,xxlat = np.meshgrid(xlong,xlat)
        	del xlong, xlat
        	nc.close()
	for it in range(0,bt.StormLon[:,iS].shape[0],1):	
		if bt.Time[it,iS] > datetime(1800,1,1,0):
	           distance = np.empty(xxlong.shape,dtype=float)
        	   er = 6371.0 #km
        	   londis = 2*np.pi*er*np.cos(xxlat/180*np.pi)/360
        	   dx = londis*(xxlong - bt.StormLon[it,iS])
        	   dy = 110 * (xxlat - bt.StormLat[it,iS])
        	   distance = np.sqrt(dx*dx+dy*dy)
		   (j0,i0) = np.unravel_index(np.argmin(distance),distance.shape)

		   var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],PIVmax),0,500
		   bt.PIwspdMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
		   bt.PIwspd[it,iS] = var[j0,i0]

		   var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],u),200,800
		   bt.UShearMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
		   bt.UShear[it,iS] = var[j0,i0]

		   var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],v),200,800
		   bt.VShearMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
		   bt.VShear[it,iS] = var[j0,i0]

		   var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],meanrh1),200,800
		   bt.rh500_300Mean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
		   bt.rh500_300[it,iS] = var[j0,i0]

		   var,radius1,radius2 = copy.copy(ldmask),0,300
		   var[var==0] = -1.
		   var[var==3] = 0.0
		   bt.landmaskMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
		   bt.landmask[it,iS] = var[j0,i0] 

	return iS
			
def getStormTranslation(lon,lat,time):
        '''
        This function find the storm speed and direction.
        lon: longitude
        lat: latitude
        time: time
        sdir: (ndarray) storm direction
        speed: (ndarray) speed
        '''
        er = 6371.0 #km
        timeInt=[]
        lonInt=[]
        latInt=[]


        for iN in range(time.shape[0]-1):
               timeInt.append(time[iN])
               lonInt.append(lon[iN])
               latInt.append(lat[iN])
               #delt = (time[iN+1]-time[iN]).seconds/60/60
               #if delt ==0:
               #   print time
               delt = 6
               inv = 1./np.float(delt)
               #if time[iN+1]-time[iN] == timedelta(hours=delt):
               for iM in range(1,delt,1):
                      timeInt.append(time[iN]+timedelta(hours=iM))
                      lonInt.append((1.-iM*inv)*lon[iN]+iM*inv*lon[iN+1])
                      latInt.append((1.-iM*inv)*lat[iN]+iM*inv*lat[iN+1])


        speed = np.zeros(lon.shape[0],dtype=float)+float('nan')
        sdir = np.zeros(lon.shape[0],dtype=float)+float('nan')
        count = 0
        for it in time:
            nup = argminDatetime(it+timedelta(hours=3),timeInt)
            ndn = argminDatetime(it-timedelta(hours=3),timeInt)
            londis = 2*np.pi*er*np.cos(latInt[nup]/180*np.pi)/360
            dlon = lonInt[nup]-lonInt[ndn]
            dlat = latInt[nup]-latInt[ndn]
            dx = londis*(lonInt[nup]-lonInt[ndn])
            dy = 110*(latInt[nup]-latInt[ndn])
            distance = np.sqrt(dx*dx+dy*dy) #km
            sdir[count]=np.arctan2(dlat,dlon)
            speed[count]=distance*1000./(nup-ndn+1)/60/60 #m/s
            count+= 1

        return sdir,speed

def getSpeedDir(iiS,block_id=None):
    '''
    This function finds the speed and direction of the storm based on 
    '''
    iS = np.int(iiS.mean(keepdims=True))
    if (bt.StormLat[:,iS]==bt.StormLat[:,iS]).any():
        it1 = np.argwhere(bt.StormLat[:,iS]==bt.StormLat[:,iS])[0,0]
        it2 = np.argwhere(bt.StormLat[:,iS]==bt.StormLat[:,iS])[-1,0]
        if it2 - it1 >=2:
           bt.trDir[it1:it2,iS],bt.trSpeed[it1:it2,iS]=\
             getStormTranslation(bt.StormLon[it1:it2,iS],\
             bt.StormLat[it1:it2,iS],bt.Time[it1:it2,iS])
    return iS
def calCoefficient_water_guess(ty1, ty2, ih):
    """
    calculate first guess coefficient from OLS
    ty1 = initial year
    ty2 = final year
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

