#/usr/bin/env python
import numpy as np
#import geopy
import copy
from netCDF4 import Dataset
from datetime import datetime,timedelta
#from geopy.distance import VincentyDistance

class bstinfo(object):
	"""
	A function to read in the /home/clee/data/NHC_ATL_bst_1581_2012.nc	
	In addition to output the storm-info variables, it outputs also empty 
        variables arrays for predictors.  
	"""
        def __init__(self,ncFileName):
                nc = Dataset(ncFileName,'r')
                self.StormId        = nc.variables['StormId'][:]
                self.StormYear      = nc.variables['StormYear'][:]
                self.StormInitMonth = nc.variables['StormInitMonth'][:]
                self.StormLon       = nc.variables['Longitude'][:][:]
                self.StormLat       = nc.variables['Latitude'][:][:]
                self.StormMwspd     = nc.variables['MaxWSPD'][:][:]
                self.StormMslp      = nc.variables['MinSLP'][:][:]
                yyyy                = nc.variables['StormYYYY'][:][:]
                mm                  = nc.variables['StormMM'][:][:]
                dd                  = nc.variables['StormDD'][:][:]
                hh                  = nc.variables['StormHH'][:][:]
                nc.close()
                dummy               = yyyy*mm*dd*hh
                self.Time           = np.empty(yyyy.shape,dtype=object)
		self.Time[:]	    = datetime(1800, 1, 1, 0)
                self.PIwspd         = np.empty(yyyy.shape,dtype=float)*float('Nan')
                self.PIslp          = np.empty(yyyy.shape,dtype=float)*float('Nan')
                self.PIwspdMean     = np.empty(yyyy.shape,dtype=float)*float('Nan')
                self.dPIwspd        = np.empty(yyyy.shape,dtype=float)*float('Nan')
                self.PIslpMean      = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.UShearMean	    = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.UShear         = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.VShearMean	    = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.VShear         = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.div200Mean	    = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.div200         = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.T200Mean	    = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.T200           = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.rh500_300Mean  = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.rh500_300      = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.rhMean	    = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.rh             = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.T100Mean	    = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.T100           = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.dThetaEMean    = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.dThetaE        = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.dThetaEsMean   = np.empty(yyyy.shape,dtype=float)*float('Nan')	
		self.dThetaEs       = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.landmask       = np.empty(yyyy.shape,dtype=float)*float('Nan')
		self.landmaskMean   = np.empty(yyyy.shape,dtype=float)*float('Nan')
                self.trSpeed        = np.zeros(yyyy.shape)
                self.trDir          = np.zeros(yyyy.shape)
                self.dVdt           = np.zeros(yyyy.shape)
                self.dPdt           = np.zeros(yyyy.shape)
                self.dVdt[self.dVdt==0]='NaN'
                self.dVdt[self.dVdt==0]='NaN'
                self.dVdt[2::,:]=self.StormMwspd[2::,:]-self.StormMwspd[0:-2,:]
                self.dPdt[1::,:]=self.StormMwspd[1::,:]-self.StormMwspd[0:-1,:]

		yyyy=np.array(yyyy,dtype='int')
		mm=np.array(mm,dtype='int')
		dd=np.array(dd,dtype='int')
		hh=np.array(hh,dtype='int')
                for ix in range(0,yyyy.shape[0],1):
                    for iy in range(0,yyyy.shape[1],1):
                        if ((dummy[ix,iy]==dummy[ix,iy]) and (dd[ix,iy]!=0)) :
                           self.Time[ix,iy] = datetime(int(yyyy[ix,iy]),int(mm[ix,iy]),int(dd[ix,iy]),int(hh[ix,iy]))

class read_ibtracs(object):
      """
	a function read /crunch/c1/clee/tracks/Allstorms.ibtracs_all.v03r08.nc
	atl - NHC ATL
	enp - NHC ENP
	wnp,sh,ni - JTWC for the rest: SH, IO, and WPC
	'global', it is reading all data
        
        lon is change to the range from 0 to 360

	to do task: use xarray to have better time calculations
	data = xr.to_nedcdf(filename)
      """
      def __init__(self,ncFileName,basins):
        #ncFileName = '/crunch/c1/clee/tracks/Allstorms.ibtracs_all.v03r08.nc'
        nc = Dataset(ncFileName,'r', format='NETCDF3_CLASSIC')
	sourceN = []
	if 'atl' in basins:
	    sourceN.append(0)
	elif 'wnp' in basins:
	    sourceN.append(10)
	elif 'sh' in basins:
	    sourceN.append(9)
	elif 'ni' in basins:
	    sourceN.append(13)			
	elif 'enp' in basins:
	    sourceN.append(15)
	elif 'global' in basins:
	    sourceN = np.array([0,9,10,13,15])
	sourceN = np.array(sourceN)
	print sourceN,basins
	if sourceN.size == 1:
           lon = nc.variables['source_lon'][:,:,sourceN[0]].T
           lat = nc.variables['source_lat'][:,:,sourceN[0]].T
           wspd = nc.variables['source_wind'][:,:,sourceN[0]].T
           days = nc.variables['source_time'][:,:].T
	   names = nc.variables['name'][:,:].T
	   stormID = nc.variables['numObs'][:]
	   dist2land = nc.variables['dist2land'][:,:].T
	else:
	   count = 0
           for isource in sourceN:
               if count == 0:
                  lon = nc.variables['source_lon'][:,:,isource].T
                  lat = nc.variables['source_lat'][:,:,isource].T
                  wspd = nc.variables['source_wind'][:,:,isource].T
                  days = nc.variables['source_time'][:,:].T
	          names = nc.variables['name'][:,:].T
		  stormID = nc.variables['numObs'][:]
	          dist2land = nc.variables['dist2land'][:,:].T
               else:
                  lon = np.hstack([lon,nc.variables['source_lon'][:,:,isource].T])
                  lat = np.hstack([lat,nc.variables['source_lat'][:,:,isource].T])
                  wspd = np.hstack([wspd,nc.variables['source_wind'][:,:,isource].T])
                  days = np.hstack([days,nc.variables['source_time'][:,:].T])
                  names = np.hstack([names,nc.variables['name'][:,:].T])
		  dist2land = np.hstack([dist2land,nc.variables['dist2land'][:,:].T])
                  stormID = np.hstack([stormID,nc.variables['numObs']])
   	       count += 1
        nNaN = np.argwhere(np.nanmax(np.array(lon),axis=0)!=-30000.).ravel()
        lon = lon[:,nNaN]
        lat = lat[:,nNaN]
        wspd = wspd[:,nNaN]
        days = days[:,nNaN]
	names = names[:,nNaN]
	stormID = stormID[nNaN]
	dist2land = dist2land[:,nNaN]
        year = np.zeros(wspd.shape[1])
        month = np.zeros(wspd.shape[1])
        count = 0
	print nNaN.shape
        for i in range(nNaN.shape[0]):
            year[i] = (datetime(1858,11,17,0,0)+timedelta(days=np.array(days)[0,i])).year
            month[i] = (datetime(1858,11,17,0,0)+timedelta(days=np.array(days)[0,i])).month
        a = np.nanmax(wspd,axis=0)
        arg = np.argwhere(a==a)[:,0]
        lon = np.array(lon[:,arg])
        lat = np.array(lat[:,arg])
        year = year[arg]
        month = month[arg]
        wspd = np.array(wspd[:,arg])
        days = np.array(days[:,arg])
        dist2land = np.array(dist2land[:,arg])
        wspd[wspd==-9990.] = np.float('nan')
        lon[lon==-30000.] = np.float('nan')
        lat[lat==-30000.] = np.float('nan')
	stormID = np.array(stormID[arg])
	names = np.array(names[:,arg])
	lon[lon<0]=lon[lon<0]+360
	self.wspd = wspd
	self.lon = lon
	self.lat = lat
	self.year = year
	self.days = days
	self.stormID = stormID
	self.names = names
	self.dist2land = dist2land
	self.month = month
class read_ibtracs_v4(object):
      """
	a function read /data2/clee/bttracks/IBTrACS.ALL.v04r00.nc
	netCDF4 library is required
	we use all USA agency - usa_lat, usa_lon, etc 
	atl - NHC ATL
	enp - NHC ENP
	wnp,sh,ni - JTWC for the rest: SH, IO, and WPC
	'global', it is reading all data
        
        lon is change to the range from 0 to 360

	to do task: use xarray to have better time calculations
	data = xr.to_nedcdf(filename)
      """
      def __init__(self,ncFileName,basins,gap):
        #ncFileName = '/data2/clee/bttracks/IBTrACS.ALL.v04r00.nc'
        nc = Dataset(ncFileName,'r', format='NETCDF3_CLASSIC')
	sourceN = []
	if 'atl' in basins:
	   sourceN.extend(['NA'])
	if 'wnp' in basins:
	   sourceN.extend(['WP'])
	if 'sh' in basins:
	   sourceN.extend(['SA','SP','SI'])
	if 'ni' in basins:
	   sourceN.extend(['NI'])
	if 'enp' in basins:
	   sourceN.extend(['EP'])
	if 'global' in basins:
	   sourceN = ['NA','WP','SA','SP','SI','NI','EP']
	sourceN = np.array(sourceN)
	arg=[]
	for i in range(sourceN.shape[0]):
	    arg.extend(np.argwhere((nc.variables['basin'][:,0,0]==sourceN[i][0])&
			(nc.variables['basin'][:,0,1]==sourceN[i][1])).ravel().tolist()) 
	arg = np.array(arg)
	print sourceN,basins
	#print(arg)
	lon = nc.variables['usa_lon'][:][arg,::gap].T
	lat = nc.variables['usa_lat'][:][arg,::gap].T
	wspd = nc.variables['usa_wind'][:][arg,::gap].T
	days = nc.variables['time'][:][arg,::gap].T
	names = nc.variables['name'][:][arg,:].T
	stormID = nc.variables['number'][:][arg]
	dist2land = nc.variables['dist2land'][:][arg,::gap].T
	trspeed = nc.variables['storm_speed'][:][arg,::gap].T
	trdir = nc.variables['storm_dir'][:][arg,::gap].T
	year = nc.variables['season'][:][arg]
	times = nc.variables['iso_time'][:][arg,::gap].T


        nNaN = np.argwhere(np.nanmax(np.array(lon),axis=0)!=-9999.).ravel()
        lon = lon[:,nNaN]
        lat = lat[:,nNaN]
        wspd = wspd[:,nNaN]
        days = days[:,nNaN]
        times = times[:,:,nNaN]
	names = names[:,nNaN]
	stormID = stormID[nNaN]
	dist2land = dist2land[:,nNaN]
	trspeed = trspeed[:,nNaN]
	trdir = trdir[:,nNaN]
	year = year[nNaN]

        a = np.nanmax(wspd,axis=0)
        arg = np.argwhere(a==a)[:,0]
        lon = np.array(lon[:,arg])
        lat = np.array(lat[:,arg])
        year = year[arg]
        wspd = np.float_(np.array(wspd[:,arg]))
        days = np.array(days[:,arg])
        dist2land = np.array(dist2land[:,arg])
        wspd[wspd==-9999.] = np.float('nan')
        lon[lon==-9999.] = np.float('nan')
        lat[lat==-9999.] = np.float('nan')
	stormID = np.array(stormID[arg])
	names = np.array(names[:,arg])
	trspeed = np.array(trspeed[:,arg])
	trdir = np.array(trdir[:,arg])
	times = np.array(times[:,:,arg])

	lon[lon<0]=lon[lon<0]+360
	self.wspd = wspd
	self.lon = lon
	self.lat = lat
	self.year = year
	self.days = days
	self.stormID = stormID
	self.names = names
	self.dist2land = dist2land
	self.year = year
	self.trspeed = trspeed
	self.trdir = trdir
	self.times = times
