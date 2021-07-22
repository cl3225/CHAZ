#!/usr/bin/env python
import numpy as np
import subprocess
import pickle
import copy
import random
from tools.util import int2str,find_range

def get_E1(bt):
    """
    get errors and residual of errors from predictand bt
    return errors(E1), residual of errors(delta1), and ranges of v0 (cat1)
    """

    NS = np.argwhere((bt['StormYear'][:]>=1981)&(bt['StormYear'][:]<=2012))[:,0]
    #data = bt['errors'][:,NS,0]
    #data2 = bt.trueY[:,NS,0]
    data = bt['errors'][:,NS]
    data2 = bt['StormMwspd'][:,NS]

    for it in range(0,data.shape[1],1):
        data1 = data[:,it]
        data4 = data2[:,it]
        data4 = data4[data1==data1]
        data1 = data1[data1==data1]
        if 'E1' not in locals():
            E1 = data1[0:data1.shape[0]]
            E0 = data1[0:data1.shape[0]]
            v0E = data4[0:data4.shape[0]]
        else:
            E1 = np.hstack([E1,data1[0:data1.shape[0]]])
            E0 = np.hstack([E0,data1[0:data1.shape[0]]])
            v0E = np.hstack([v0E,data4[0:data4.shape[0]]])
    E2 = E0*E1*v0E
    E0 = E0[E2==E2]
    E1 = E1[E2==E2]
    v0E = v0E[E2==E2]
    c11,c0 = np.polyfit(E0,E1,1)

    c1 = v0E.min()
    c2 = v0E.max()
    for iC in np.arange(v0E.min(),v0E.max(),10):
        if v0E[v0E<iC].shape[0] > 50: c1 = iC; break
    for iC in np.arange(v0E.max(),v0E.min(),-10):
        if v0E[v0E>iC].shape[0] > 50: c2 = iC; break
    cat1 = np.arange(c1,c2+10,10) # range for Vinit

    return E0,v0E,cat1

def get_E1_delta1(bt):
    """
    get errors and residual of errors from predictand bt
    return errors(E1), residual of errors(delta1), and ranges of v0 (cat1)
    """

    iS1 = np.argwhere(bt['StormYear'][:]==1980)[0][0]
    iS2 = np.argwhere(bt['StormYear'][:]==1999)[-1][0]
    data = bt['errors'][:][:,iS1:iS2,1]
    data2 = bt['trueY'][:,:][:,iS1:iS2,0]

    for it in range(0,data.shape[1],1):
        data1 = data[:,it]
        data4 = data2[:,it]
        data4 = data4[data1==data1]
        data1 = data1[data1==data1]
        if 'E1' not in locals():
            E1 = data1[1:data1.shape[0]]
            E0 = data1[0:data1.shape[0]-1]
            v0E = data4[1:data4.shape[0]]
        else:
            E1 = np.hstack([E1,data1[1:data1.shape[0]]])
            E0 = np.hstack([E0,data1[0:data1.shape[0]-1]])
            v0E = np.hstack([v0E,data4[1:data4.shape[0]]])
    E2 = E0*E1*v0E
    E0 = E0[E2==E2]
    E1 = E1[E2==E2]
    v0E = v0E[E2==E2]
    c11,c0 = np.polyfit(E0,E1,1)

    for it in range(0,data.shape[1],1):
        data1 = data[:,it]
        data4 = data2[:,it]
        data4 = data4[data1==data1]
        data1 = data1[data1==data1]
        data3 =  data1[1:data1.shape[0]]-\
              (data1[0:data1.shape[0]-1]*c11)
        data4 = data4[1:]
        if 'delta1' not in locals():
           delta1 = data3[1:data3.shape[0]]
           delta0 = data3[0:data3.shape[0]-1]
           v0delta = data4[1:data4.shape[0]]
        else:
           delta1 = np.hstack([delta1,data3[1:data3.shape[0]]])
           delta0 = np.hstack([delta0,data3[0:data3.shape[0]-1]])
           v0delta = np.hstack([v0delta,data4[1:data4.shape[0]]])
    delta3 = delta1*delta0*v0delta
    delta1 = delta1[delta3==delta3]
    delta0 = delta0[delta3==delta3]
    v0delta = v0delta[delta3==delta3]

    c1 = v0delta.min()
    c2 = v0delta.max()
    for iC in np.arange(v0delta.min(),v0delta.max(),10):
        if v0delta[v0delta<iC].shape[0] > 50: c1 = iC; break
    for iC in np.arange(v0delta.max(),v0delta.min(),-10):
        if v0delta[v0delta>iC].shape[0] > 50: c2 = iC; break
    cat1 = np.arange(c1,c2+10,10) # range for Vinit

    return E0,v0E,delta1,cat1,v0delta,c11

def get_mean(meanFile,predictors):
    '''
    calculate mean in X and Y based on predictors
    '''
    hour = range(12,132,12)
    n2 = len(predictors)
    meanX = np.empty([len(hour),n2],dtype=float)+np.float('nan')
    meanY = np.empty([len(hour)],dtype=float)+np.float('nan')
    stdX  = np.empty([len(hour),n2],dtype=float)+np.float('nan')
    stdY  = np.empty([len(hour)],dtype=float)+np.float('nan')
    f = open(meanFile)
    b = f.readlines()
    f.close()
    count1 = 0
    for ih in hour:
        lm = b[ih/12].rstrip().rsplit()
        meanX[count1,:] = lm[3:3+n2]
        meanY[count1]  = lm[1]
        stdX[count1,:] = lm[3+n2::]
        stdY[count1]   = lm[2]
        count1 += 1
    return meanX,meanY,stdX,stdY

def findError(errort,v0E,v0,cat1):
    '''
    calculate error
    '''
    if (v0 == v0):
       if v0 <= cat1.min():
          j0 = 0
          error1 = errort[v0E<cat1[j0+1]]
       elif v0 >= cat1.max():
          j0 = len(cat1)-1
          error1 = errort[v0E>=cat1[j0]]
       else:
          [j0,j1] = find_range(cat1,v0)
          error1 = errort[((v0E>=cat1[j0]) & (v0E<cat1[j0+1]))]
       error = random.choice(error1)
    else:
       error = 0

    return error

