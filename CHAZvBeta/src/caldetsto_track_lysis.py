#!/usr/bin/env python
###
# adding lysis
# adding hard-threshold for non-WPC storms to have higher initial storm intensity. 
####
import numpy as np
import pickle
import dask.array as da
import sys
import gc
import copy
import pandas as pd
import tools.module_stochastic as module_sto
from tools.readbst import read_ibtracs
import time
from tools import util
import random
from datetime import datetime
from netCDF4 import Dataset,date2num
from module_riskModel import fst2bt
from tools.util import int2str
from dask.diagnostics import ProgressBar
import Namelist as gv
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tools.readbst import read_ibtracs
import numpy.ma as ma
import netCDF4 as nc
from netCDF4 import Dataset
import tools.regression4 as reg4 

jetcmap = plt.cm.get_cmap("jet",31)
jet_vals = jetcmap(np.arange(31))
cmap1 = mcolors.LinearSegmentedColormap.from_list('newjet', jet_vals)
colors = jet_vals
colorcat = np.arange(0,155,5) 
xbin,ybin,basinMap = util.getbasinMap()

def get_determin(iiS,block_id=None):
    '''
    This function finds the deterministic component of the autoregressive TC intensity model.
    iiS: bt data for intensity model
    '''
    iS = np.int(iiS.mean(keepdims=True))
    #iS  = iiS
    lT = 2
    dummy = bt.StormLon[:,iS][bt.StormLon[:,iS]==bt.StormLon[:,iS]]
    TimeDepends = ['dThetaEsMean','T200Mean','rhMean','rh500_300Mean','div200Mean']
    if ((dummy.any()) and (np.abs(bt.StormLat[0,iS])>=5.)):
       it1 = 0
       it2 = np.min([np.argwhere(bt.StormLon[:,iS]==dummy[-1])[-1,0],bt.StormLon.shape[0]-5])
       if ((bt.StormLat[0,iS] >=0) and (bt.StormLon[0,iS]>=120) and (bt.StormLon[0,iS]<=180)):
          bt.determin[0,iS] = np.max([20,random.choice(intV[intV==intV])]) # kts 
       else:
          bt.determin[0,iS] = np.max([25,random.choice(intV[intV==intV])]) # kts 
       #bt.determin[0,iS] = intV[iS] # kts 
       dvdt = 0.0 
       ih = 12
       lT = np.int(ih/6) ### track model formate is every 12 hours
       v0 = bt.determin[0,iS]
       for it in range(it1,it2+2,2):
          if ((it+lT<bt.StormLon.shape[0]) and (bt.StormLon[it,iS]==bt.StormLon[it,iS])\
                and (bt.landmaskMean[it,iS]==bt.landmaskMean[it,iS]) \
                and (bt.landmaskMean[it+lT,iS]==bt.landmaskMean[it+lT,iS]) and not np.isnan(v0)):
             if(((bt.landmaskMean[it+lT,iS]<= -0.5)&(bt.landmaskMean[it,iS]<= -0.5))):
                 predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2']
                 result,meanX,meanY,stdX,stdY = np.copy(result_w),np.copy(meanX_w),\
                                                np.copy(meanY_w),np.copy(stdX_w),np.copy(stdY_w)
             elif(((bt.landmaskMean[it+lT,iS]> -0.5)|(bt.landmaskMean[it,iS]> -0.5))):
                 predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
                 result,meanX,meanY,stdX,stdY = np.copy(result_l),np.copy(meanX_l),\
                                                np.copy(meanY_l),np.copy(stdX_l),np.copy(stdY_l)
             h1,v1 = reg4.getPrediction_v0input_result\
                     (bt,meanX,meanY,stdX,stdY,it,\
                     iS,[ih],result,predictors,\
                     TimeDepends,v0,dvdt)
             v1 = v1[1]
             if v1 < 10:
                v1 = np.float('Nan') 
                break;
             bt.determin[it+lT,iS] = v1
             if ((bt.determin[it1:it+lT:lT,iS].max()>35) and (bt.determin[it,iS]<=35) and (bt.determin[it-lT,iS]<=35)):
                break;
             dvdt = v1-v0
             v0 = v1
       for iit in range(it1+1,np.min([it2+1,bt.StormLon.shape[0]-2]),2):
           a = bt.determin[iit-1,iS]*bt.determin[iit+1,iS]
           if a==a:
              bt.determin[iit,iS] = \
              0.5*(bt.determin[iit-1,iS]+bt.determin[iit+1,iS])
    return iS

def get_stochastic(iiS,block_id=None):
    '''
    This function finds the stochastic component of the autoregressive TC intensity model.
    iiS: bt data for intensity model
    '''
    TimeDepends = ['dThetaEsMean','T200Mean','rhMean','rh500_300Mean','div200Mean']
    iS = np.int(iiS.mean(keepdims=True))
    dummy = bt.StormLon[:,iS][bt.StormLon[:,iS]==bt.StormLon[:,iS]]
    if ((dummy.any()) and (np.abs(bt.StormLat[0,iS])>=5.)):
       it1 = 0
       it2 = np.min([np.argwhere(bt.StormLon[:,iS]==dummy[-1])[-1,0],bt.StormLon.shape[0]-5])
       if ((bt.StormLat[0,iS] >=0) and (bt.StormLon[0,iS]>=120) and (bt.StormLon[0,iS]<=180)):
          bt.stochastic[0,iS,iNN] = np.max([20,random.choice(intV[intV==intV])]) # kts 
       else:
          bt.stochastic[0,iS,iNN] = np.max([25,random.choice(intV[intV==intV])]) # kts 
       ih = 12
       lT = np.int(ih/6)
       v0 = bt.stochastic[0,iS,iNN]
       dvdt = 0.
       for it in range(it1,it2+2,2):
          if ((it+lT<bt.StormLon.shape[0]) and (bt.StormLon[it,iS]==bt.StormLon[it,iS])\
                and (bt.landmaskMean[it,iS]==bt.landmaskMean[it,iS]) \
                and (bt.landmaskMean[it+lT,iS]==bt.landmaskMean[it+lT,iS]) and not np.isnan(v0)):
             if(((bt.landmaskMean[it+lT,iS]<= -0.5)&(bt.landmaskMean[it,iS]<= -0.5))):
                 predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2']
                 result,meanX,meanY,stdX,stdY = result_w,meanX_w,\
                                                meanY_w,stdX_w,stdY_w
             elif(((bt.landmaskMean[it+lT,iS]> -0.5)|(bt.landmaskMean[it,iS]> -0.5))):
                 predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
                 result,meanX,meanY,stdX,stdY = result_l,meanX_l,\
                                                meanY_l,stdX_l,stdY_l
             h1,v1 = reg4.getPrediction_v0input_result\
                     (bt,meanX,meanY,stdX,stdY,it,\
                     iS,[ih],result,predictors,\
                     TimeDepends,v0,dvdt)
             error = module_sto.findError(E0,v0E,v0,cat1)
             bt.error[it,iS,iNN] = error 
             v1 = v1[1]
             v1 = v1-error
             if v1 < 10:
                break;
             #if ((it >= it1+4*lT) and (bt.stochastic[it,iS,iNN]<=35) and (bt.stochastic[it-lT,iS,iNN]<=35)):
             bt.stochastic[it+lT,iS,iNN] = v1
             if ((bt.stochastic[it1:it+lT:lT,iS,iNN].max()>35) and (bt.stochastic[it,iS,iNN]<=35) and (bt.stochastic[it-lT,iS,iNN]<=35)):
                break;
             dvdt = v1-v0
             v0 = v1
       for iit in range(it1+1,np.min([it2+1,bt.StormLon.shape[0]-2]),2):
           a = bt.stochastic[iit-1,iS,iNN]*bt.stochastic[iit+1,iS,iNN]
           if a==a:
              bt.stochastic[iit,iS,iNN] = \
              0.5*(bt.stochastic[iit-1,iS,iNN]+bt.stochastic[iit+1,iS,iNN])


    return iS

def calIntensity(EXP,iy,ichaz):
    '''
    This function calculates the intesity using an autoregressive model (Lee et al. (2015, 2016a)).
    EXP: location of historical simulations defined in Namelist.py
    iy: year of current iteration in CHAZ.py
    ichaz: current ensemble iteration in CHAZ.py`
    '''
    ipath2 = gv.ipath+gv.Model+'/HIST/'
    global result_w, meanY_w, stdX_w, meanX_w, stdY_w
    global result_l, meanY_l, stdX_l, meanX_l, stdY_l

    ### read coefficient from reanalysis data trainning,
    ### as well as recording the mean and STD
   # with open (gv.opath,'rb') as f:
#	 bt = pickle.load(f)
#	 result_w = pickle.load(f) 
#	 meanX_w_obs = pickle.load(f)
#	 stdX_w_obs = pickle.load(f)
#	 meanY_w_obs = pickle.load(f)
#	 stdY_w_obs = pickle.load(f)
#	 result_l = pickle.load(f) 
#	 meanX_l_obs = pickle.load(f)
#	 stdX_l_obs = pickle.load(f)
#	 meanY_l_obs = pickle.load(f)
#	 stdY_l_obs = pickle.load(f)
#	 f.close()
    bt2 = nc.Dataset(gv.opath,'r')
    result_wi = nc.Dataset(gv.ipath + 'result_w.nc', 'r')
    result_w = ma.getdata(result_wi['params'][:])
    result_li = nc.Dataset(gv.ipath + 'result_l.nc','r')
    result_l = ma.getdata(result_li['params'][:])
    observed_data = nc.Dataset(gv.ipath + 'observed_data.nc', 'r')
    meanX_w_obs = ma.getdata(observed_data['meanX_w_obs'][:])
    stdX_w_obs = ma.getdata(observed_data['stdX_w_obs'][:])
    meanY_w_obs = ma.getdata(observed_data['meanY_w_obs'][:])
    stdY_w_obs = ma.getdata(observed_data['stdY_w_obs'][:])
    meanX_l_obs = ma.getdata(observed_data['meanX_l_obs'][:])
    stdX_l_obs = ma.getdata(observed_data['stdX_l_obs'][:])
    meanY_l_obs = ma.getdata(observed_data['meanY_l_obs'][:])
    stdY_l_obs = ma.getdata(observed_data['stdY_l_obs'][:])

    NS = np.argwhere((bt2['StormYear'][:]>=1981)&(bt2['StormYear'][:]<=2012))[:,0]
    global E0, v0E, cat1, intV
    intV = bt2['StormMwspd'][:][0,NS]
    E0, v0E,cat1, = module_sto.get_E1(bt2)
    del bt2

    #read the synthetic tracks
    global bt
    with open(gv.output_path+'trackPredictorsbt'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','r')as f:
	 bt = pickle.load(f)
	 f.close()
    ### read mean and std from the HIST data
    #with open (gv.ipath+'/coefficient_meanstd.pik','r') as f:
#	 dummy = pickle.load(f)
#	 meanX_w = pickle.load(f)
#	 stdX_w = pickle.load(f)
#	 meanY_w = pickle.load(f)
#	 stdY_w = pickle.load(f)
#	 dummy = pickle.load(f)
#	 meanX_l=pickle.load(f)
#	 stdX_l = pickle.load(f)
#	 meanY_l = pickle.load(f)
#	 stdY_l = pickle.load(f)
#    f.close()
    coeff_meanstd = nc.Dataset(gv.pre_path+'coefficient_meanstd.nc','r')
    meanX_w = ma.getdata(coeff_meanstd['meanX_w'][:])
    stdX_w = ma.getdata(coeff_meanstd['stdX_w'][:])
    meanY_w = ma.getdata(coeff_meanstd['meanY_w'][:])
    stdY_w = ma.getdata(coeff_meanstd['stdY_w'][:])
    meanX_l = ma.getdata(coeff_meanstd['meanX_l'][:])
    stdX_l = ma.getdata(coeff_meanstd['stdX_l'][:])
    meanY_l = ma.getdata(coeff_meanstd['meanY_l'][:])
    stdY_l = ma.getdata(coeff_meanstd['stdY_l'][:])

    nsto = gv.CHAZ_Int_ENS 
    bt.__dict__['determin'] = np.zeros(bt.StormLon.shape)*np.float('nan')
    bt.__dict__['stochastic'] = np.zeros([bt.StormLon.shape[0],bt.StormLon.shape[1],nsto])*np.float('nan')
    bt.__dict__['error'] = np.zeros([bt.StormLon.shape[0],bt.StormLon.shape[1],nsto])*np.float('nan')

    NS = np.arange(0,bt.StormYear.shape[0],1)
    #NNS1,NNS2 = np.meshgrid(NS,np.arange(0,nsto,1))
    #nnS1 = da.from_array(NNS1.ravel().astype(dtype=np.int32),chunks=(1,))
    #nnS2 = da.from_array(NNS2.ravel().astype(dtype=np.int32),chunks=(1,))
    nS = da.from_array(NS.astype(dtype=np.int32),chunks=(1,))  

    new = da.map_blocks(get_determin,nS,chunks=(1,), dtype=nS.dtype)
    with ProgressBar():
         n = new.compute(scheduler='synchronous',num_workers=5)
    del new
    gc.collect()
    #for iS in NS:
    #    print iS
    #    b = get_determin(np.array([iS, iS]))
    #print 'down calculate deterministic'

    new =da.map_blocks(get_stochastic,nS,chunks=(1,), dtype=nS.dtype)
    global iNN
    for iNN in range(nsto):
	time1 = time.time()
	with ProgressBar():
	     n = new.compute(scheduler='synchronous',num_workers=5)
	gc.collect()
	#print iNN,time.time()-time1
    del new

   # with open (gv.output_path+'bt_stochastic_det'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','w') as f:
    #     pickle.dump(bt,f)
    #f.close()

    for iens in range(0,1):
        count1 = 0
        if count1 ==0:
            bt1 = bt
            maxCol = bt.PIslp.shape[0]
            count1 = 1
        else:
            for iv in dir(bt):
                if (('__' not in iv) and ('Time' not in iv) and (getattr(bt,iv).ndim==2)):
                    b = np.zeros([maxCol,getattr(bt,iv).shape[1]])*np.float('nan')
                    b[0:getattr(bt,iv).shape[0],:] = getattr(bt,iv)
                    setattr(bt,iv,b)
                elif (('__' not in iv) and ('Time' not in iv) and ('timeY' not in iv) and (getattr(bt,iv).ndim==3)):
                    b = np.zeros([maxCol,getattr(bt,iv).shape[1],getattr(bt,iv).shape[2]])*np.float('nan')
                    b[0:getattr(bt,iv).shape[0],:,:] = getattr(bt,iv)
                    setattr(bt,iv,b)
                elif 'Time' in iv:
                    b = np.empty([maxCol,getattr(bt,iv).shape[1]],dtype=object)
                    b[:] = datetime(1800, 1, 1, 0)
                    b[0:getattr(bt,iv).shape[0],:] = getattr(bt,iv)
                    setattr(bt,iv,b)
            for iv in dir(bt1):
                if (('__' not in iv) and ('predictors' not in iv) and ('timeY' not in iv)):
                    b = np.hstack([getattr(bt1,iv), getattr(bt,iv)])
                    setattr(bt1,iv,b)
            del bt
        #### get basin-information
        lon0,lat0 = bt1.StormLon[0,:],bt1.StormLat[0,:]
        basin = util.defineBasin(lon0,lat0,xbin,ybin,basinMap)

        ####
        dummy = bt1.StormLon*bt1.StormLat
        id1 = np.argwhere(dummy!=dummy)[:,0]
        iS1 = np.argwhere(dummy!=dummy)[:,1]
        bt1.stochastic[id1,iS1,:] = np.float('nan')
        mask_arr_stochastic = ma.masked_invalid(bt1.stochastic)

        #### get rid of storms that move fron N.H. to S.H from SH. to NH, from ATL to ENP
        iS1 = np.argmax(mask_arr_stochastic,axis=0)
        basin1 = np.array([util.defineBasin(bt1.StormLon[iS1[nn],nn],\
                  bt1.StormLat[iS1[nn],nn],xbin,ybin,basinMap) \
                  for nn in range(iS1.shape[0])])
        basin0 = np.tile(basin,(40,1)).T
        arg_last = np.array([[pd.Series(bt1.stochastic[:,nn,iN]).last_valid_index()\
                    for nn in range(bt1.StormLon.shape[1])] for iN in range(40)])
        arg_last[arg_last == np.array(None)] = 0
        arg_last = arg_last.astype(None)
        arg_last = np.int_(arg_last).T
        basin2 = np.array([util.defineBasin(bt1.StormLon[arg_last[nn],nn],\
                   bt1.StormLat[arg_last[nn],nn],xbin,ybin,basinMap) \
                   for nn in range(arg_last.shape[0])])
        iS1, iN1 = np.where((((basin0<5)&(basin2>=5))|((basin0>=5)&(basin2<5))|\
                    ((basin0==0)&(basin2==1))))
        bt1.stochastic[:,iS1,iN1] = np.float('nan')
        iS1,iN1 = np.where((((basin0<5)&(basin1>=5))|((basin0>=5)&(basin1<5))|\
                    ((basin0==0)&(basin1==1))))
        bt1.stochastic[:,iS1,iN1] = np.float('nan')            
        
        lon0, lat0 = bt1.StormLon[0,:],bt1.StormLat[0,:]
        basin = util.defineBasin(lon0,lat0,xbin,ybin,basinMap)

        #### get rid of storms  that has never developed
        bt1.stochastic[bt1.stochastic==0] = np.float('nan')
        max5 = np.nanmax(bt1.stochastic[0:21,:,:],axis=0)
        iS1 = np.argwhere(max5<35)[:,0]
        iN1 = np.argwhere(max5<35)[:,1]
        bt1.stochastic[:,iS1,iN1] = np.float('nan')
        maxall = np.nanmax(bt1.stochastic,axis=0)
        v0 = bt1.stochastic[0,:,:]
        #### get rid of storms that becoming unstable
        iS1 = np.argwhere(maxall>300)[:,0]
        iN1 = np.argwhere(maxall>300)[:,1]
        bt1.stochastic[:,iS1,iN1] = np.float('nan')
        #### get rid of storms that initially formed with 2 degree lon,lat
        iS1 = np.argwhere(np.abs(bt1.StormLat[0,:])<2)[:,0]
        bt1.stochastic[:,iS1,:] = np.float('nan')
        bt1.StormLon[0,iS1] = np.float('nan')
        
        #### get rid of storms that are not from within the range
        iS1 = np.argwhere(basin==0)
        bt1.StormLon[0,iS1] = np.float('nan')
        
        #### get rid of storms that have no intensity record
        maxall = np.nanmax(bt1.stochastic,axis=0)
        #changed
        iS1 = np.argwhere(maxall<=0)[:,0]
        iN1 = np.argwhere(maxall<=0)[:,1]
        bt1.stochastic[:,iS1,iN1] = np.float('nan')
        #bt1.stochastic[:,basin!=1,:] = np.float('nan')
        arg = np.argwhere(bt1.StormLon[0,:] == bt1.StormLon[0,:])[:][:,0]
        
        newlon = bt1.StormLon[:,arg]
        newlat = bt1.StormLat[:,arg]
        newwspd = bt1.stochastic[:,arg]
        newdatenum = bt1.Time[:,arg]
        newYear = bt1.StormYear[arg]
        newMonth = bt1.StormInitMonth[arg]
        
        #for iN in range(20,21):
         #   tracks={'xlong':[],'xlat':[],'v':[],'color':[],'color_n':[],'year':[],'month':[],'day':[],'hour':[]}
          #  maxloc={'xlong':[],'xlat':[],'vmax':[]}
            #arg = np.argwhere((basin_vmax[:,iN]==1)&(basin==1)).ravel()
           # tracks,maxloc = \
            #  get_tracks_maxloc(bt,bt.stochastic[:,:,iN],
             # tracks,maxloc,colorcat,colors)
            #tracks = module_vmaxMap.get_colorCode(tracks,colorcat,colors)

            #plot the spaghetti plots for Obs
            #var = tracks['v'].tolist()
            #titleName = 'ATL '+int2str(iy,4)+'_ens'+int2str(iens,3) 
            #figName='MLR'
            #tracks1 = tracks

            #lon1, lon2, londis = 240,360,30
            #lat1, lat2, latdis = 0,50,15
            #MyFigsize, MyFontsize = (3.5,2),10
            #plcbar = True
            #fig = module_vmaxMap.plt_spaghetti(tracks1,titleName,lon1,lon2,lat1,lat2,MyFigsize,MyFontsize,londis,latdis,jetcmap,colorcat,colors,plcbar)
            #plt.savefig(gv.Model+'_ATL_'+int2str(iy,4)+'_ens'+int2str(iens,3)+'_int_'+int2str(iN,2)+'.png',dpi=300)

        #print('newdatenum:', newdatenum.shape)
        #print('newwspd:', newwspd.shape)
        #print('arg:',arg.shape)
        ### netcdf via xarray
        #times = pd.date_range(start='1950-01-01 00:00', freq:'H',periods=arg.shape[0])

        #dummy dataframe
        ensembleNum = np.arange(newwspd.shape[2])
        stormID = np.arange(arg.shape[0])
        lifelength = np.arange(newdatenum.shape[0])

        ds = xr.Dataset({
          'longitude': xr.DataArray(
                         data = newlon,
                         dims = ['lifelength','stormID'],
                         coords = {'lifelength':lifelength, 'stormID':stormID},
                         attrs = {
                             '_FillValue': np.float('nan'),
                             'units': 'degrees east'
                             }
                         ),
           

          'latitude': xr.DataArray(
                         data = newlat,
                         dims = ['lifelength','stormID'],
                         coords = {'lifelength':lifelength, 'stormID':stormID},
                         attrs = {
                             '_FillValue': np.float('nan'),
                             'units'     : 'degrees north'
                             }
                         ),
         
          'Mwspd': xr.DataArray(
                         data =np.rollaxis(newwspd,2,0),
                         dims = ['ensembleNum','lifelength','stormID'],
                         coords = {'ensembleNum':ensembleNum,'lifelength':lifelength, 'stormID':stormID},
                         attrs = {
                             '_FillValue': np.float('nan'),
                             'units'     : 'kt'
                             }
                         ),
          'year': xr.DataArray(
                         data = newYear,
                         dims = ['stormID'],
                         coords = {'stormID':stormID},
                         attrs = {
                             'units'     : 'year'
                             }
                         ), 
          'time': xr.DataArray(
                         data = date2num(newdatenum, units='days since 1950-01-01 00:00', calendar='standard'),
                         dims = ['lifelength','stormID'],
                         coords = {'lifelength':lifelength, 'stormID':stormID},
                         attrs = {
                             'units'     : 'days since 1950-01-01 00:00'
                             }
                         )
                }
            )

        file_name = gv.output_path+gv.Model+'_'+int2str(iy,4)+'_ens'+int2str(iens,3)+'.nc'       
        print(file_name)
        ds.to_netcdf(file_name)

    
    return()	




