#!/usr/bin/env python
from numpy import *
from datetime import datetime,timedelta
from netCDF4 import Dataset
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bisect import bisect_left, bisect_right
import scipy
import copy
er = 6371.0 #km

def date_interpolation(dateInput,fileInput):
    """
    fileInput is a 3-d gloabl fields  with time-axis on the first column, eg. fileInput[it,ix,iy]
    the function return fields interpolated linearly to date on dateInput.
    if dateInput.month is December, then it returns Dec. field
    """
    if dateInput.day >= 15:
       if dateInput.month < 12:
          date0 = datetime(dateInput.year,dateInput.month,15,0,0)
          date1 = datetime(dateInput.year,dateInput.month+1,15,0,0) 
          dfdays = (dateInput-date0).days
          xp = [0, (date1-date0).days]
          ratio = np.interp(dfdays,xp,[0,1])
          fileOutput = fileInput[dateInput.month-1,:,:]*(1-ratio)+fileInput[dateInput.month,:,:]*(ratio)
       else:
          fileOutput= fileInput[dateInput.month-1,:,:]
    else:
       if dateInput.month == 1:
          fileOutput= fileInput[dateInput.month-1,:,:]
       else:
          date0 = datetime(dateInput.year,dateInput.month-1,15,0,0)
          date1 = datetime(dateInput.year,dateInput.month,15,0,0)
          dfdays = (dateInput-date0).days
          xp = [0, (date1-date0).days]
          ratio = np.interp(dfdays,xp,[0,1])
          fileOutput = fileInput[dateInput.month-2,:,:]*(1-ratio)+fileInput[dateInput.month-1,:,:]*ratio
    return fileOutput



def find_range(array, a):
    """ 
    Return two indices in which a is in between in an array
    array has to be a monotonic increase or decrease array
    """
    if (diff(array)<0).any(): 
       start = bisect_right(array[::-1],a)
       end = bisect_left(array[::-1],a)
       end = array.shape[0]-start
       start = end
    else:
       start = bisect_right(array,a)
       end = bisect_left(array,a)
    return (start-1, end)





def int2str(num,l):
    """
    Given an integer num and desired length l, returns the str(num)
    with appended leading zeroes to match the size l.
    """
    if len(str(num))<l:
        return (l-len(str(num)))*'0'+str(num)
    else:
        return str(num)

def num2str(num,precision):
    """
    Given an float number and desired precision number precision, returm
    str(num)	
    """ 
    return "%0.*f"%(precision,num)



def argminDatetime(time0,time):
    """
    Returns the index of datetime array time for which
    the datetime time0 is nearest.
    """
    time = np.array(time)
    n0 = 0
    delt = np.abs(time[-1]-time[0])
    for n in range(time.size):
        if np.abs(time0-time[n]) < delt:
            delt = np.abs(time0-time[n])
            n0 = n
    return n0


def virtualTemperature(T,r):
    """
    """
    return T*(0.622+r)/(0.622*(1+r))


def getTextFile(filename):
    """
    Returns a list where each element contains text from each line 
    of given text file.
    """
    return [line.rstrip() for line in open(filename,'r').readlines()]



def squeeze(a):
    "squeeze(a) returns a with any ones from the shape of a removed"
    a = asarray(a)
    b = asarray(a.shape)
    return reshape (a, tuple (compress (not_equal (b, 1), b)))

def getbasinMap():
    '''
    This function returns x bins, y bins, and a plot of the basin map.
    '''
    xbin, ybin = np.arange(0,365,5),np.arange(-90,95,5)
    xcenter = 0.5*(xbin[0:-1]+xbin[1:])
    ycenter = 0.5*(ybin[0:-1]+ybin[1:])
    basinMap = np.zeros([xcenter.shape[0],ycenter.shape[0]])
    lonc,latc = np.meshgrid(xcenter,ycenter)
    n_sin = np.argwhere((lonc<90) & (latc<0)&(latc>=-45))
    n_aus = np.argwhere((lonc>=90)&(lonc<160)&(latc<0)&(latc>=-45))
    n_spc = np.argwhere((lonc>=160)&(lonc<240)&(latc<0)&(latc>=-45))
    n_ni = np.argwhere((lonc<100) & (latc>=0)&(latc<=45))
    n_wnp = np.argwhere((lonc>=100)&(lonc<180)&(latc>=0)&(latc<=45))
    n_enp = np.argwhere(((lonc>=180)&(lonc<235)&(latc>=0)&(latc<=45)))
    n_atl = np.argwhere((lonc>=235)&(latc>=0)&(latc<=45))

    basinMap[n_atl[:,1],n_atl[:,0]]=1
    basinMap[n_enp[:,1],n_enp[:,0]]=2
    basinMap[n_wnp[:,1],n_wnp[:,0]]=3
    basinMap[n_ni[:,1],n_ni[:,0]]=4
    basinMap[n_sin[:,1],n_sin[:,0]]=5
    basinMap[n_aus[:,1],n_aus[:,0]]=6
    basinMap[n_spc[:,1],n_spc[:,0]]=7
    a= np.arange(xcenter.shape[0]*ycenter.shape[0]).reshape(xcenter.shape[0],ycenter.shape[0])
    basinMap[a==1716] = 2
    basinMap[a==1715] = 2
    basinMap[a==1714] = 2
    basinMap[a==1713] = 2
    basinMap[a==1712] = 2
    basinMap[a==1711] = 2
    basinMap[a==1710] = 2
    basinMap[a==1751] = 2
    basinMap[a==1750] = 2
    basinMap[a==1749] = 2
    basinMap[a==1748] = 2
    basinMap[a==1747] = 2
    basinMap[a==1746] = 2
    basinMap[a==1787] = 2
    basinMap[a==1786] = 2
    basinMap[a==1785] = 2
    basinMap[a==1784] = 2
    basinMap[a==1783] = 2
    basinMap[a==1782] = 2
    basinMap[a==1822] = 2
    basinMap[a==1821] = 2
    basinMap[a==1820] = 2
    basinMap[a==1819] = 2
    basinMap[a==1818] = 2
    basinMap[a==1854] = 2
    basinMap[a==1855] = 2
    basinMap[a==1856] = 2
    basinMap[a==1857] = 2
    basinMap[a==1890] = 2
    basinMap[a==1891] = 2
    basinMap[a==1892] = 2
    basinMap[a==1893] = 2
    basinMap[a==1926] = 2
    basinMap[a==1927] = 2
    basinMap[a==1928] = 2
    basinMap[a==1962] = 2
    basinMap[a==1963] = 2
    basinMap[a==1964] = 2
    #basinMap[a==1680] = 7
    #basinMap[a==1681] = 7
    #basinMap[a==1683] = 7
    basinMap[a==1714] = 2
    basinMap[a==1718] = 2
    basinMap[a==1719] = 2
    basinMap[a==1716] = 2
    basinMap[a==1717] = 2
    basinMap[a==1753] = 2
    basinMap[a==1752] = 2
    basinMap[a==1754] = 2
    basinMap[a==1755] = 2
    basinMap[a==1785] = 2
    basinMap[a==1786] = 2
    basinMap[a==1787] = 2
    basinMap[a==1821] = 2
    basinMap[a==1822] = 2
    basinMap[a==1823] = 2
    basinMap[a==1752] = 2
    basinMap[a==1753] = 2
    basinMap[a==1754] = 2
    basinMap[a==1755] = 2
    basinMap[a==1785] = 2
    basinMap[a==1786] = 2
    basinMap[a==1787] = 2
    basinMap[a==1821] = 2
    basinMap[a==1822] = 2
    basinMap[a==1823] = 2
    basinMap[a==1752] = 2
    basinMap[a==1753] = 2
    basinMap[a==1998] = 2
    basinMap[a==1999] = 2
    return xbin,ybin,basinMap
def defineBasin(lon0_obs,lat0_obs,basinlon,basinlat,basinMap):
    '''
    Defines basin based on observations and knowledge about the basin.
    lon0_obs: initial longitude observed
    lat0_obs: initial latitude observed
    basinlon: basin longitude
    basinlat: basin latitude 
    basinMap: map of basin
    '''
    lat0_obs[lat0_obs>=90] = 89.9
    x = np.floor(lon0_obs/np.diff(basinlon)[0])
    y = np.floor((lat0_obs-basinlat[0])/np.diff(basinlat)[0])
    basin = basinMap[np.int_(x),np.int_(y)]
    return(basin)
def getStormTranslation(lon,lat,time):
        '''
        This function finds the storm's translation - speed and direction.
        lon:(ndarray) longitude
        lat: (ndarray) latitude
        time: (ndarray) times
        '''
        timeInt=[]
        lonInt=[]
        latInt=[]

        for iN in range(time.shape[0]-1):
               timeInt.append(time[iN])
               lonInt.append(lon[iN])
               latInt.append(lat[iN])
               delt = (time[iN+1]-time[iN]).seconds/60/60
               #print delt, lon[iN],lon[iN+1]
               if ((lon[iN+1] ==lon[iN+1]) and (delt >0)) :
                   inv = 1./np.float(delt)
                   for iM in range(1,delt,1):
                      timeInt.append(time[iN]+timedelta(hours=iM))
                      lonInt.append((1.-iM*inv)*lon[iN]+iM*inv*lon[iN+1])
                      latInt.append((1.-iM*inv)*lat[iN]+iM*inv*lat[iN+1])
               else:
                   timeInt.append(datetime(1800,1,1,0,0))
                   lonInt.append(np.float('nan'))
                   latInt.append(np.float('nan'))


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
            speed[count]=distance*1000./(nup-ndn)/60/60 #m/s
            count+= 1

        return sdir,speed

