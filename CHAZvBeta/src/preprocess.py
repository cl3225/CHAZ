#!/usr/bin/env python
import numpy as np
import sys
import os
import gc
import time
import Namelist as gv
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from tools.util import int2str
from netCDF4 import Dataset 
def createNetCDF(covMatrix,iy,xlong,xlat):
	'''
	This function writes wind covariance into netcdf file.
        covMatrix: wind covariance matrix
        iy: year in iteration
        xlong: longitude points
        xlat: latitude points
        Returns netCDF file
	'''
	# write wind covariance netcdf
        var = ['u200p2D','v200p2D','u850p2D','v850p2D']
        nc = Dataset(gv.pre_path+'Cov1_'+int2str(iy,4)+'.nc','w',format='NETCDF3_CLASSIC')
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
                
                var1 = nc.createVariable(vname,np.dtype('float32').char,('month','latitude','longitude'))
                var1.units = '(ms-1)^2'
                var1[:] = covMatrix[count,:,:,:]
                count += 1
        nc.close()
        return()




def run_preProcesses():
	y1 = gv.Year1
	y2 = gv.Year2
	monthly_csv = gv.monthlycsv
	with open(monthly_csv,'r') as f:
		df_mary = f.readlines()
	df_sub_uam = []
	df_sub_vam = []
	df_sub_hurm = []
	for ia in df_mary:
		if 'u_component' in ia and 'daily' not in ia:
			df_sub_uam.append(ia[:-1])
		if 'v_component' in ia and 'daily' not in ia:
			df_sub_vam.append(ia[:-1])
		if 'relative_humidity' in ia:
			df_sub_hurm.append(ia[:-1])
	df_sub_uam = np.array(df_sub_uam)
	df_sub_vam = np.array(df_sub_vam)
	df_sub_hurm = np.array(df_sub_hurm)
	modely1m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1][:4] for i in range(df_sub_uam.shape[0])])
	modely2m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1].split('-')[-1][:4] for i in range(df_sub_uam.shape[0])])
	dummy = xr.open_dataset(df_sub_uam[0])
	olon = dummy.longitude.values
	olat = dummy.latitude.values
	ollon,ollat = np.meshgrid(olon,olat) #original grids
	a  = loadmat(gv.pre_path+'PI_'+str(y1)+'.mat')
	#print(a)
	xlat = a['Y'][:,0]
	xlon = a['X'][:,0]
	xllon,xllat = np.meshgrid(xlon,xlat) # desired grids

	## first gives 4 empty arrays for saving the closest 4 origional grids indicies
	## at each desired grids based on distance
	time1 = time.time()
	jl1,jl2 = np.zeros(xllon.shape),np.zeros(xllon.shape)
	il1,il2 = np.zeros(xllon.shape),np.zeros(xllon.shape)
	for jx in range (xllon.shape[0]):
		for ix in range(xllon.shape[1]):
			#print jx, ix
			dis = np.sqrt((ollon-xllon[jx,ix])**2+(ollat-xllat[jx,ix])**2)
			ndis = np.argwhere(dis==dis.min())[-1]
			if xllat[jx,ix] < ollat[ndis[0],ndis[1]]: # higher then the desiered grid 
				jl1[jx,ix] = ndis[0]-1
				jl2[jx,ix] = ndis[0]
			else: # lower
				jl1[jx,ix] = ndis[0]
				jl2[jx,ix] = ndis[0]+1
			if xllon[jx,ix] < ollon[ndis[0],ndis[1]]: # further right
				il1[jx,ix] = ndis[1]-1
				il2[jx,ix] = ndis[1]
			else: # further left
				il1[jx,ix] = ndis[1]
				il2[jx,ix] = ndis[1]+1
	jl1 = np.int_(jl1)
	jl2 = np.int_(jl2)
	il1 = np.int_(il1)
	il2 = np.int_(il2)
	il2[il2==olon.shape[0]]=0 # periodical BL, so 360 is equal to 0 degree.
	jl1[jl1<0]=0
	jl2[jl2==olat.shape[0]]=olat.shape[0]-1
	print 'done mapping', time.time()-time1


	#### for covMatrix 
	if True:
		for iy in range(y1, y2+1):
			time1 = time.time()
			filename = gv.pre_path+'Cov_'+int2str(iy,4)+'.nc'
			nc = Dataset(filename,'r',format='NETCDF3_CLASSIC')
			covMatrix = np.zeros([10,nc.variables['month'].shape[0],xlat.shape[0],xlon.shape[0]])
			count = 0
			for iv in nc.variables.keys()[3:]:
				var = nc.variables[iv][:]
				for im in range(var.shape[0]):
					covMatrix[count,im] =\
					(var[im,jl1,il1]*(ollon[jl2,il2]-xllon)*(ollat[jl2,il2]-xllat)+\
					var[im,jl1,il2]*(xllon-ollon[jl2,il1])*(ollat[jl2,il1]-xllat)+\
					var[im,jl2,il1]*(ollon[jl1,il2]-xllon)*(xllat-ollat[jl1,il2])+\
					var[im,jl2,il2]*(xllon-ollon[jl1,il1])*(xllat-ollat[jl1,il1]))/\
					((ollon[jl2,il2]-ollon[jl2,il1])*(ollat[jl2,il2]-ollat[jl1,il2]))
				count+=1
			covMatrix[covMatrix>1000000000] = -1e+20
			createNetCDF(covMatrix,iy,xlon,xlat)
	    

	### Now we are going to iterates it through years.	
	arg_y1 = -1
	for iy in range(y1, y2+1):
		arg_ym = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel()[0]
		if arg_ym != arg_y1:
                        ds_uam = xr.open_dataset(df_sub_uam[arg_ym])
                        ds_vam = xr.open_dataset(df_sub_vam[arg_ym])
                        ds_hurm = xr.open_dataset(df_sub_hurm[arg_ym])
                        arg_p250m = np.argwhere(ds_uam.level.values==250.).ravel()[0]
                        arg_p500m = np.argwhere(ds_uam.level.values==500.).ravel()[0]
                        arg_p700m = np.argwhere(ds_uam.level.values==700.).ravel()[0]
                        arg_p850m = np.argwhere(ds_uam.level.values==850.).ravel()[0]
			arg_y1 = arg_ym
		time1 = time.time()
		arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)).ravel()
		ua850 = ds_uam.u[arg_tm,arg_p850m].values
		va850 = ds_vam.v[arg_tm,arg_p850m].values
		ua250 = ds_uam.u[arg_tm,arg_p250m].values
		va250 = ds_vam.v[arg_tm,arg_p250m].values
		hur500 = ds_hurm.r[arg_tm,arg_p500m].values
		hur700 = ds_hurm.r[arg_tm,arg_p700m].values
		hur850 = ds_hurm.r[arg_tm,arg_p850m].values
		hur = (hur500+hur700+hur850)/3.
		
		### some model the dynamic variable and thermodyanmic variable are output at grid center
		### we need to mapping them on the same grid
		if hur.shape != va250.shape:
			### latitude ###
			if hur.shape[1] != va250.shape[1]:
				ax1 = va250.shape[1]+1
				hur1 = np.zeros([hur.shape[0],ax1,hur.shape[2]])
				hur1[:,1:-1,:] =  hur
				hur1[:,0,:] = hur[:,0,:]
				hur1[:,-1,:] = hur[:,-1,:]
				del hur
				hur = 0.5*(hur1[:,0:-1,:]+hur1[:,1:,:])
				del hur1
			if hur.shape[2] != va250.shape[2]:
				ax2 = va250.shape[2]+1
				hur1 = np.zeros([hur.shape[0],hur.shape[1],ax2])
				hur1[:,:,1:-1] =  hur
				hur1[:,:,0] = hur[:,:,0]
				hur1[:,:,-1] = hur[:,:,-1]
				del hur
				hur = 0.5*(hur1[:,:,0:-1]+hur1[:,:,1:])
				del hur1
			
		a  = loadmat(gv.pre_path+'PI_'+str(iy)+'.mat')
		PI2 =np.rollaxis(a['VmaxI'],2,0)

		varName = ['hur','ua850','va850','ua250','va250','PI']
		varUnit = ['percentage','ms-1','ms-1','ms-1','ms-1','ms-1']
		for iv in varName[:-1]:
			var = eval(iv)
			var2 = np.zeros([hur.shape[0],xllon.shape[0],xllat.shape[1]])
			for im in range(hur.shape[0]):
				var2[im] = \
				(var[im,jl1,il1]*(ollon[jl2,il2]-xllon)*(ollat[jl2,il2]-xllat)+\
				var[im,jl1,il2]*(xllon-ollon[jl2,il1])*(ollat[jl2,il1]-xllat)+\
				var[im,jl2,il1]*(ollon[jl1,il2]-xllon)*(xllat-ollat[jl1,il2])+\
				var[im,jl2,il2]*(xllon-ollon[jl1,il1])*(xllat-ollat[jl1,il1]))/\
				((ollon[jl2,il2]-ollon[jl2,il1])*(ollat[jl2,il2]-ollat[jl1,il2]))

			var2[var2>1000000000] = -1e+20 
			exec(iv+'2 = var2')
			del var, var2
		NewFile = gv.pre_path+int2str(iy,4)+'_'+gv.ENS+'.nc'
		nc = Dataset(NewFile,'w',format='NETCDF3_CLASSIC')  
		nc.createDimension('Longitude',xllon.shape[1])
		Longitude = nc.createVariable('Longitude',np.dtype('float32').char,('Longitude'))
		Longitude.units = 'degree to East'
		Longitude[:] = xllon[0,:]      
		nc.createDimension('Latitude',xllon.shape[0])
		Latitude = nc.createVariable('Latitude',np.dtype('float32').char,('Latitude'))
		Latitude.units = 'degree to N'
		Latitude[:] = xllat[:,0]
		nc.createDimension('Month',12)
		Month = nc.createVariable('Month',np.dtype('int32').char,('Month'))
		Month.units=''
		Month[:] = np.int_(np.arange(1,13,1))
		count = 0
		for iv in varName:
			var = nc.createVariable(iv+'2',np.dtype('float32'),\
			      ('Month','Latitude','Longitude'),fill_value=-1e+20)
			var.units = varUnit[count]
			var[:] = eval(iv+'2') 
			count += 1
		nc.close()
		if np.mod(iy,10)==0:
			gc.collect()
			print 'year'+int2str(iy,4),time.time()-time1
	return()

def createNetCDF_A(A1,iy,im,xllon,xllat):
        """
        create NetCDF file for the syn wind, it will be 124MB each
        iy: year in iteration
        im: model in iteration
        xllon: longitude array
        xllat: latitude array
        """
        nc = Dataset(gv.pre_path+'A_'+int2str(iy,4)+int2str(im,2)+'.nc','w',format='NETCDF3_CLASSIC')
        nc.createDimension('latitude',xllat.shape[0])
        nc.createDimension('longitude',xllon.shape[0])
        nc.createDimension('days',A1.shape[1])
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
        days[:] = range(0,A1.shape[1],1)
        Adim[:] = range(0,A1.shape[0],1)
        AA = nc.createVariable('A',np.dtype('float32').char,('Adim','days','latitude','longitude'))
        AA.units = ''
        AA[:] = A1
        nc.close()
        return()


def run_preProcesses_A():
    '''
    Preprocess to create A
    '''
    ipath = gv.ipath+'/'+gv.Model+'/'
    y1, y2 = gv.Year1,gv.Year2
    ens = gv.ENS

    ### using any of the files to derive mapping indices from CMIP5 grid to 2x2 grid ###
    ### because CMIP6 and the desired grids are both rectangle grids, it is ok to do so. 
    ### we only need to do this calculate once!  
    EXP='HIST' if y1<=2005 else 'RCP85'
    #ipath1 = gv.ipath+gv.Model+'/'+EXP+'/'
    ipath1 = gv.opath
    filename = ipath1+'A_'+str(y1)+'05.nc'
    nc = Dataset(filename,'r',format='NETCDF3_CLASSIC')
    lon = nc.variables['longitude'][:]
    lat = nc.variables['latitude'][:]
    nc.close()
    llon,llat = np.meshgrid(lon,lat) #original grids

    fileName1 = gv.ipath+gv.PImodelname+'_'+EXP+'_ENS'+ens[1]+'_'+str(y1)+'.mat'
    a = loadmat(ipath1+fileName1)
    xlat1 = a['Y'][:,0]
    xlon1 = a['X'][:,0]
    xlon,xlat = np.meshgrid(xlon1,xlat1) # desired grids
    ## first gives 4 empty arrays for saving the closest 4 origional grids indicies
    ## at each desired grids based on distance
    time1 = time.time()
    jl1,jl2 = np.zeros(xlon.shape),np.zeros(xlon.shape)
    il1,il2 = np.zeros(xlon.shape),np.zeros(xlon.shape)
    for jx in range (xlon.shape[0]):
        for ix in range(xlon.shape[1]):
            dis = np.sqrt((llon-xlon[jx,ix])**2+(llat-xlat[jx,ix])**2)
            ndis = np.argwhere(dis==dis.min())[-1]
            if xlat[jx,ix] < llat[ndis[0],ndis[1]]: # higher then the desiered grid 
               jl1[jx,ix] = ndis[0]-1
               jl2[jx,ix] = ndis[0]
            else: # lower
               jl1[jx,ix] = ndis[0]
               jl2[jx,ix] = ndis[0]+1
            if xlon[jx,ix] < llon[ndis[0],ndis[1]]: # further right
               il1[jx,ix] = ndis[1]-1
               il2[jx,ix] = ndis[1]
            else: # further left
               il1[jx,ix] = ndis[1]
               il2[jx,ix] = ndis[1]+1
    jl1 = np.int_(jl1)
    jl2 = np.int_(jl2)
    il1 = np.int_(il1)
    il2 = np.int_(il2)
    il2[il2==lon.shape[0]]=0 # periodical BL, so 360 is equal to 0 degree.
    jl1[jl1<0]=0
    jl2[jl2==lat.shape[0]]=lat.shape[0]-1
    print 'done mapping', time.time()-time1

    for iy in range(y1, y2+1):
        EXP='HIST' if iy<=2005 else 'RCP85'
        ipath1 = gv.opath
        time1 = time.time()
        for im in range(1,13):
            FileAName =\
              ipath1+'A_'+str(iy)+int2str(im,2)+'.nc'
            nc = Dataset(FileAName,'r',format='NETCDF3_CLASSIC')
            iv =  nc.variables.keys()[4]
            var = nc.variables[iv][:]
            A = np.zeros([var.shape[0],var.shape[1],xlat.shape[0],xlon.shape[1]])
            for imm in range(var.shape[0]):
                for inn in range(var.shape[1]):
                    A[imm,inn,] =\
                     (var[imm,inn,jl1,il1]*(llon[jl2,il2]-xlon)*(llat[jl2,il2]-xlat)+\
                    var[imm,inn,jl1,il2]*(xlon-llon[jl2,il1])*(llat[jl2,il1]-xlat)+\
                    var[imm,inn,jl2,il1]*(llon[jl1,il2]-xlon)*(xlat-llat[jl1,il2])+\
                    var[imm,inn,jl2,il2]*(xlon-llon[jl1,il1])*(xlat-llat[jl1,il1]))/\
                    ((llon[jl2,il2]-llon[jl2,il1])*(llat[jl2,il2]-llat[jl1,il2]))
            nc.close()
            createNetCDF_A(A,iy,im,xlon1,xlat1,'./')
    return()

