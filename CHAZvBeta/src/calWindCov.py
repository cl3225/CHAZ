#!/usr/bin/env python
import numpy as np
from datetime import datetime as real_datetime
from netcdftime import utime,datetime
import sys
import subprocess
from tools.util import int2str
from scipy.interpolate import interp1d
from calendar import monthrange
from netCDF4 import Dataset
import gc
import Namelist as gv
import pandas as pd
import xarray as xr

def createNetCDF(covMatrix,iy,xlong,xlat):
        var = ['u200p2D','v200p2D','u850p2D','v850p2D']
        nc = Dataset(gv.pre_path+'Cov_'+int2str(iy,4)+'.nc','w',format='NETCDF3_CLASSIC')
        nc.createDimension('latitude',xlat.shape[0])
        nc.createDimension('longitude',xlong.shape[0])
        nc.createDimension('month',covMatrix.shape[1])
        lats = nc.createVariable('latitude',np.dtype('float32').char,('latitude',))
        lons = nc.createVariable('longitude',np.dtype('float32').char,('longitude',))
        months = nc.createVariable('month',np.dtype('int32').char,('month',))
        lats.units = 'degrees_north'
        lons.units = 'degrees_east'
        months.units = 'month_in_year'
        lats[:] = xlat
        lons[:] = xlong
        months[:] = range(1,covMatrix.shape[1]+1,1)

        ### start to create variables
        count = 0
        for iv in range(len(var)):
            for iiv in range(iv,len(var),1):
                vname = var[iv][:-2]+var[iiv][:-2]
                #print vname
                var1 = nc.createVariable(vname,np.dtype('float32').char,('month','latitude','longitude'))
                var1.units = '(ms-1)^2'
                var1[:] = covMatrix[count,:,:,:]
                count += 1
        nc.close()
        return()

def fillinNaN(var,neighbors):
	"""
	replacing masked area using interpolation 
	"""
	for ii in range(var.shape[0]):
		a = var[ii,:,:]
		count = 0
		while np.any(a.mask):
			a_copy = a.copy()
			for hor_shift,vert_shift in neighbors:
			    if not np.any(a.mask): break
			    a_shifted=np.roll(a_copy,shift=hor_shift,axis=1)
			    a_shifted=np.roll(a_shifted,shift=vert_shift,axis=0)
			    idx=~a_shifted.mask*a.mask
			    #print count, idx[idx==True].shape 
			    a[idx]=a_shifted[idx]
			count+=1
		var[ii,:,:] = a
	return var

def my_cov(x,y,naxis):
	n = x.shape[naxis]
	cov_bias = np.mean(x*y,axis=naxis)-(np.mean(x,axis=naxis)*np.mean(y,axis=naxis))
	cov_bias = cov_bias*n/(n-1)
	return cov_bias

def run_windCov():
	##### reading monthly data through a list
	##### daily_csv should be in the namelist.py
	y1 = gv.Year1
	y2 = gv.Year2
	monthly_csv = gv.monthlycsv
	with open(monthly_csv,'r') as f:
		df_mary = f.readlines()
	df_sub_uam = []
	df_sub_vam = []
	df_sub_hurm = []
	df_sub_uad = []
	df_sub_vad = []
	for ia in df_mary:
		if 'u_component' in ia and 'daily' not in ia:
			df_sub_uam.append(ia[:-1])
		if 'u_component' in ia and 'dailymean' in ia:
			df_sub_uad.append(ia[:-1])
		if 'v_component' in ia and 'daily' not in ia:
			df_sub_vam.append(ia[:-1])
		if 'v_component' in ia and 'dailymean' in ia:
			df_sub_vad.append(ia[:-1])
		if 'relative_humidity' in ia:
			df_sub_hurm.append(ia[:-1])
	df_sub_uam = np.array(df_sub_uam)
	df_sub_vam = np.array(df_sub_vam)
	df_sub_hurm = np.array(df_sub_hurm)
	df_sub_uad = np.array(df_sub_uad)
	df_sub_vad = np.array(df_sub_vad)
	modely1m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1][:4] for i in range(df_sub_uam.shape[0])])
	modely2m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1].split('-')[-1][:4] for i in range(df_sub_uam.shape[0])])

	##### reading daily data through Lamont URL
	##### daily_csv should be in the namelist.py
	modely1d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])
	modely2d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])

	arg_y1 = -1
	neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
	for iy in range(gv.Year1,gv.Year2+1):
		arg_yd = np.argwhere((modely1d<=iy)&(modely2d>=iy)).ravel()
		arg_ym = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel()[0]
		#arg_ym0 = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel().tolist()
		#arg_ym1 = np.argwhere((modely1m<=iy+1)&(modely2m>=iy+1)).ravel().tolist()
		#arg_ym2 = np.argwhere((modely1m<=iy-1)&(modely2m>=iy-1)).ravel().tolist()
		#arg_ym0.extend(arg_ym1)
		#arg_ym0.extend(arg_ym2)
		#arg_ym = np.unique(np.array(arg_ym0))
		if arg_yd[0] != arg_y1:
			ds_uam = xr.open_dataset(df_sub_uam[arg_ym])
			ds_vam = xr.open_dataset(df_sub_vam[arg_ym])
			arg_p250m = np.argwhere(ds_uam.level.values==250.).ravel()[0]
			arg_p850m = np.argwhere(ds_uam.level.values==850.).ravel()[0]
			ds_uad = xr.open_mfdataset(df_sub_uad[arg_yd])
			ds_vad = xr.open_mfdataset(df_sub_vad[arg_yd])
			arg_p250d = np.argwhere(ds_vad.level.values==250.).ravel()[0]
			arg_p850d = np.argwhere(ds_vad.level.values==850.).ravel()[0]
			xlong = ds_vam.longitude.values
			xlat = ds_vam.latitude.values
			dsvadtime = pd.to_datetime(np.arange(ds_vad.day.shape[0]),unit='D',origin=pd.Timestamp(str(iy)+'-01-01'))
		###### This is a better way but xr.interP does no support chunk in the interpolation axis
		#arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)|(ds_uam.time.dt.year.values==iy-1)|(ds_uam.time.dt.year.values==iy+1)).ravel()
		arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)).ravel()
		arg_td = np.argwhere((dsvadtime.year.values==iy)).ravel()
		### cftime.DatetimeNoLeap  #### THIS IS REALLY for CESM2's calendar?? 
		missing_value = 1e+20

		date_daily = dsvadtime[arg_td]
		date_monthly =ds_uam.time[arg_tm]
		m = np.argwhere(date_daily.day==15).ravel() ## roughly middle of the month
		ua250m = ds_uam.u[arg_tm,arg_p250m].values
		ua850m = ds_uam.u[arg_tm,arg_p850m].values
		ua850m[ua850m!=ua850m] = 1e+20
		ua850m = np.ma.masked_values(ua850m, missing_value)
		ua850m = fillinNaN(ua850m,neighbors)
		va250m = ds_vam.v[arg_tm,arg_p250m].values
		va850m = ds_vam.v[arg_tm,arg_p850m].values
		va850m[va850m!=va850m] = 1e+20
		va850m = np.ma.masked_values(va850m, missing_value)
		va850m = fillinNaN(va850m,neighbors)
		d = np.arange(0,date_daily.shape[0])
		f = interp1d(m,ua250m,bounds_error=False,fill_value="extrapolate",axis=0)
		ua250md = f(d)
		f = interp1d(m,va250m,bounds_error=False,fill_value="extrapolate",axis=0)
		va250md = f(d)
		f = interp1d(m,ua850m,bounds_error=False,fill_value="extrapolate",axis=0)
		ua850md = f(d)
		f = interp1d(m,va850m,bounds_error=False,fill_value="extrapolate",axis=0)
		va850md = f(d)
		##### now we are going to do the monthly cov.
		covMatrix = np.zeros([10,12,va250md.shape[1],va250md.shape[2]])
		for im in range(1,13):
			arg_td = np.argwhere((dsvadtime.year==iy)&(dsvadtime.month==im)).ravel()
			ua250d = ds_uad.u[arg_td,arg_p250d].values
			ua850d = ds_uad.u[arg_td,arg_p850d].values
			va250d = ds_vad.v[arg_td,arg_p250d].values
			va850d = ds_vad.v[arg_td,arg_p850d].values
			#missing_values = nc.variables['ua'].missing_value
			missing_value = 1e+20
			ua850d[ua850d!=ua850d] = 1e+20
			va850d[va850d!=va850d] = 1e+20
			va850d = np.ma.masked_values(va850d, missing_value)
			ua850d = np.ma.masked_values(ua850d, missing_value)
			ua850d = fillinNaN(ua850d,neighbors)
			va850d = fillinNaN(va850d,neighbors)

			arg_md = np.argwhere((dsvadtime[arg_td].year==iy)&(dsvadtime[arg_td].month==im)).ravel()
			u250p = ua250d-ua250md[arg_md]
			u250p2D = u250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			u850p = ua850d-ua850md[arg_md]
			u850p2D = u850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			v250p = va250d-va250md[arg_md]
			v250p2D = v250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			v850p = va850d-va850md[arg_md]
			v850p2D = v850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			var = ['u250p2D','v250p2D','u850p2D','v850p2D']
			count = 0
			for iv in range(len(var)):
				vname = var[iv]
				for iiv in range(iv,len(var),1):
					vname1 = var[iiv]
					covMatrix[count,im-1,:,:] = \
						my_cov(eval(vname),eval(vname1),0).reshape([u250p.shape[1],u250p.shape[2]])
						#np.hstack([np.cov(eval(vname)[:,igrid],eval(vname1)[:,igrid])[0,1] \
						#for igrid in range(u250p2D.shape[1])]).reshape([u250p.shape[1],u250p.shape[2]])
					count += 1
			print iy, im, count, vname, vname1

		createNetCDF(covMatrix,iy,xlong,xlat)
		del covMatrix,u250p, u250p2D, u850p, u850p2D, v250p, v250p2D, v850p, v850p2D  
	    	gc.collect()
