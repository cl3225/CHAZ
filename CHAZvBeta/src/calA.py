#!/usr/bin/env python

import numpy as np
import random
import time
import copy
import subprocess
import gc
import sys
from calendar import monthrange
from scipy import linalg
from datetime import datetime
from tools.util import int2str, date_interpolation
from netCDF4 import Dataset,date2num
import Namelist as gv

def createNetCDF(A1,iy,im,xllon,xllat):
        """
        crerate NetCDF file for the syn wind, it will be 124MB each
        """
        nc = Dataset(gv.pre_path+'A_'+int2str(iy,4)+int2str(im,2)+'.nc','w',format='NETCDF3_CLASSIC')
        nc.createDimension('latitude',xllat.shape[0])
        nc.createDimension('longitude',xllon.shape[0])
        nc.createDimension('days',A1.shape[3])
        nc.createDimension('Adim',A1.shape[0])
        lats = nc.createVariable('latitude',np.dtype('float32').char,('latitude',))
        lons = nc.createVariable('longitude',np.dtype('float32').char,('longitude',))
        days = nc.createVariable('days',np.dtype('int32').char,('days',))
        Adim = nc.createVariable('Adim',np.dtype('int32').char,('Adim',))
        lats.units = 'degrees_north'
        lons.units = 'degrees_east'
        days.units = 'days_in_month'
        Adim.units = 'dimensions_in_A'
        lats[:] = xllat
        lons[:] = xllon
        days[:] = range(0,A1.shape[3],1)
        Adim[:] = range(0,A1.shape[0],1)
        AA = nc.createVariable('A',np.dtype('float32').char,('Adim','days','latitude','longitude'))
        AA.units = ''
        AA[:] = np.rollaxis(A1,3,1)
        nc.close()
        return()

def run_calA():
	for iy in range(gv.Year1, gv.Year2+1):
		filename = gv.pre_path+ 'Cov1_'+int2str(iy,4)+'.nc'
		nc = Dataset(filename,'r',format='NETCDF3_CLASSIC')
		xllon = nc.variables['longitude'][:]
		xllat = nc.variables['latitude'][:]
		var = ['u200p','v200p','u850p','v850p']
		cov = {}
		for iv in range(len(var)):
			for iiv in range(iv,len(var),1):
				vname = var[iv]+var[iiv]
				cov[vname] = nc.variables[vname][:,:,:]
				cov[vname][:,0,:] = cov[vname][:,1,:]
				cov[vname][:,-1,:] = cov[vname][:,-2,:]
		nc.close()

		for im in range(1,13,1):
			time1 = time.time()
			nday = monthrange(iy,im)[1]
			### for each month has one F, this is where the randomness from
			cov2d = {}
			for iv in range(len(var)):
				for iiv in range(iv,len(var),1):
					vname = var[iv]+var[iiv]
					vname1 = var[iv]+var[iiv]
					cov2d[vname1] = np.dstack([date_interpolation(datetime(iy,im,iday,0,0),cov[vname1]) for iday in range(1,nday+1,1)])

			time1 = time.time()

			a = np.array(cov2d[cov2d.keys()[0]].shape).tolist()
			covm2d = np.zeros([10]+a)
			count1 = 0
			for iv in range(len(var)):
				for iiv in range(iv,len(var),1):
					vname = var[iv]+var[iiv]
					covm2d[count1,:,:,:] = cov2d[vname][:,:,:]
					count1 += 1

			covv = np.zeros([4,4]+a)
			covv[0,:,:,:,:] = covm2d[0:4,:,:,:]
			covv[:,0,:,:,:] = covm2d[0:4,:,:,:]
			covv[1,1:,:,:,:] = covm2d[4:7,:,:,:]
			covv[1:,1,:,:,:] = covm2d[4:7,:,:,:]
			covv[2,2:,:,:,:] = covm2d[7:9,:,:,:]
			covv[2:,2,:,:,:] = covm2d[7:9,:,:,:]
			covv[3,3,:,:,:] = covm2d[9,:,:,:]
			#print 'finish interp cov data', time.time()-time1
			time1 = time.time()

			A = np.zeros(covv.shape)
			A1 = np.zeros([10]+a)
			count = 0
			for iday in range(nday):
				for ixx in range(xllon.shape[0]):
					for iyy in range(xllat.shape[0]):
						if np.isnan(covv[:,:,iyy,ixx,iday]).any():
							print 'break'
							break
						A[:,:,iyy,ixx,iday] = linalg.cholesky(covv[:,:,iyy,ixx,iday]).T

			A1[0,:,:,:] = A[0,0,:,:,:]
			A1[1,:,:,:] = A[1,0,:,:,:]
			A1[2,:,:,:] = A[1,1,:,:,:]
			A1[3,:,:,:] = A[2,0,:,:,:]
			A1[4,:,:,:] = A[2,1,:,:,:]
			A1[5,:,:,:] = A[2,2,:,:,:]
			A1[6,:,:,:] = A[3,0,:,:,:]
			A1[7,:,:,:] = A[3,1,:,:,:]
			A1[8,:,:,:] = A[3,2,:,:,:]
			A1[9,:,:,:] = A[3,3,:,:,:]
			#print 'finish cal A data', time.time()-time1
			time1 = time.time()
			createNetCDF(A1,iy,im,xllon,xllat)
			#print 'finish saving A data', time.time()-time1
			gc.collect()
	return()
