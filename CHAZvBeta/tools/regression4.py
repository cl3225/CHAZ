import numpy as np
import time
import subprocess
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from tools.util import int2str,find_range
from numpy import nanmean
from matplotlib.dates import date2num,HourLocator,DayLocator,DateFormatter

class Vregression1():
        """
        creates regression object
        """
        def __init__(self,n1,n2):
                self.y=np.empty([n1],dtype=float)*float('nan')
                self.x=np.empty([n1,n2],dtype=float)*float('nan')


def writePredictors(bt,leadingTime,ty1,ty2,py1,py2,predictors,fstType):
        """
	a function write the predictors into ascii fies
        """

	
        lT = np.int(leadingTime/6)
        n1 = bt.StormMslp.shape[0]*bt.StormMslp.shape[1]
        n2 = len(predictors)
        varRegress = Vregression1(n1,n2)
        varRegress1 = Vregression1(n1,n2)
        TimeDepends = ['dThetaEMean','T200Mean','rhMean','rh500_300Mean','div200Mean','T100Mean','dThetaEsMean'] 

        iS = 0
        count1 = 0
        while True:
	      # determining if the storm is initiated from the preiod of interested
              if ( (int(bt.StormYear[iS]) >= ty1) & (int(bt.StormYear[iS]) < ty2) ):
		 # now start from the lTth best-track time, so the initial-time is it - lT
                 for it in range(lT,bt.StormLon[:,iS].shape[0],1):
		     # not a NaN
                     if bt.StormLon[it,iS]==bt.StormLon[it,iS]:
			# the dependent valrabie is changes in Maximum wind speed, so it is the current time - initial time
                        varRegress.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS]
                        count2 = 0
			# the independent variables are from the initial time, so it is it - lT
                        for var in predictors:
                            if var in 'SHRD':
                               varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))
                            elif var in 'SHRD_I':
                               varRegress.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
                            elif var in 'dPIwspd':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS]
                            elif var in 'dPIslp':
                               varRegress.x[count1,count2] = nanmean(bt.PIslp[it-lT:it+1,iS])-bt.StormMslp[it-lT,iS]
                            elif var in TimeDepends:
                               varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,iS])
                            elif var in 'landmaskMean':
                               varRegress.x[count1,count2] = getattr(bt,var)[it,iS]-getattr(bt,var)[it-lT,iS]
                            elif var in 'StormMwspd':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])
                            elif var in 'StormMwspd2':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**2
                            elif var in 'StormMwspd3':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**3
                            elif var in 'MPI':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])
                            elif var in 'MPI2':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])**2
                            elif var in 'dPIwspd2':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**2
                            elif var in 'dPIwspd3':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**3
                            elif 'dVdt2' in var:
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**2
                            elif 'dVdt3' in var:
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**3
                            elif var in 'landmaskMean': 
                               varRegress.x[count1,count2] =  getattr(bt,var)[it,iS]-getattr(bt,var)[it-lT,iS]
                            else:
                               varRegress.x[count1,count2] = getattr(bt,var)[it-lT,iS]
                            count2 += 1
                        count1 += 1

              iS += 1
              if iS > bt.StormYear.shape[0]-1:break

        a = 1
        for iv in range(0,n2,1):
            a = a*varRegress.x[:,iv]
        a = a*varRegress.y

        X = sm.add_constant(varRegress.x[a==a,:],prepend = 'True')
        Y = varRegress.y[a==a]


        results = sm.OLS(Y,X).fit()

        ##### writing predictor files ####
        predictorFileName='predictor_'+int2str(leadingTime,3)+'_'+fstType+'.txt'
        f=open(predictorFileName,'w+')
        string = 'Y '
        for i in range(0,len(predictors),1):
            string = string+' '+ predictors[i]
        string = string+'\n'
        f.write(string)
        for i in range(0,Y.shape[0],1):
            string = ' '+repr(Y[i])
            for iv in range(1,n2+1,1):
                string =string+' '+repr(X[i,iv])

            string = string+'\n'
            f.write(string)
        f.close()
	 
	return

def calMultiRegression_coef_mean(bt,leadingTime,ty1,ty2,predictors,fstType):
	"""
        calculate mean coefficients for multiple regression
        bt: best track data
        ty1: year1
        ty2: year2
        predictors: predictors
        fstType: fst variable type
	"""
	import copy
	
	lT = np.int(leadingTime/6)
        n1 = bt.StormMslp.shape[0]*bt.StormMslp.shape[1]
        n2 = len(predictors)
        varRegress = Vregression1(n1,n2)
        varRegress1 = Vregression1(n1,n2)
        TimeDepends = ['dThetaEMean','T200Mean','rhMean','rh500_300Mean','div200Mean','T100Mean','dThetaEsMean'] 


	iS = bt.StormYear.tolist().index(ty1) 
        count1 = 0
	while True:
	      if ( (int(bt.StormYear[iS]) >= ty1) & (int(bt.StormYear[iS]) < ty2) ):
		 for it in range(lT,bt.StormLon[:,iS].shape[0],1):
                     if((bt.StormLon[it,iS]==bt.StormLon[it,iS])&(bt.landmaskMean[it,iS]!=0)):
                        if (fstType == 'Vmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS] 	
                        elif (fstType == 'dVmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
			count2 = 0
                        for var in predictors:
                            if var in 'SHRD':
                               varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))
			    elif var in 'SHRD_I':
                               varRegress.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
			    elif var in 'dPIwspd':
			       varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS]	
			    elif var in 'dPIslp':
			       varRegress.x[count1,count2] = nanmean(bt.PIslp[it-lT:it+1,iS])-bt.StormMslp[it-lT,iS]	
			    elif var in TimeDepends:
                               varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,iS])
                            else:
                               varRegress.x[count1,count2] = getattr(bt,var)[it-lT,iS]
                            count2 += 1
                        count1 += 1 
 
	      iS += 1
              if iS > bt.StormYear.shape[0]-1:break
	print varRegress1.x[0,:]

        a = 1
	for iv in range(0,n2,1):
            a = a*varRegress.x[:,iv] 	
	a = a*varRegress.y

	X = sm.add_constant(varRegress.x[a==a,:],prepend = 'True')
	Y = varRegress.y[a==a]
	X,meanX,stdX,Y,meanY,stdY = normalizeTransform(X,Y) 
	results = sm.OLS(Y,X).fit()
	return results, meanX,meanY,stdX,stdY

def calMultiRegression_coef_mean_land(bt,leadingTime,ty1,ty2,predictors,fstType):
	"""
        calculate mean coefficients for multiple regression on land
        bt: best track data
        ty1: year1
        ty2: year2
        predictors: predictors
        fstType: fst variable type
	"""
	import copy
	
	lT = np.int(leadingTime/6)
        n1 = bt.StormMslp.shape[0]*bt.StormMslp.shape[1]
        n2 = len(predictors)
        varRegress = Vregression1(n1,n2)
        varRegress1 = Vregression1(n1,n2)
        TimeDepends = ['dThetaEMean','T200Mean','rhMean','rh500_300Mean','div200Mean','T100Mean','dThetaEsMean'] 


	iS = bt.StormYear.tolist().index(ty1) 
        count1 = 0
	while True:
	      if ( (int(bt.StormYear[iS]) >= ty1) & (int(bt.StormYear[iS]) < ty2) ):
		 for it in range(lT,bt.StormLon[:,iS].shape[0],1):
                     #### if the starting or destination point has more than 50% land over 300km radius
                     if((bt.StormLon[it,iS]==bt.StormLon[it,iS])&((bt.landmaskMean[it,iS]> -0.5)|(bt.landmaskMean[it-lT,iS]> -0.5))):
                        if (fstType == 'Vmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS] 	
                        elif (fstType == 'dVmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
			count2 = 0
                        for var in predictors:
                            if var in 'SHRD':
                               varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))
			    elif var in 'SHRD_I':
                               varRegress.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
			    elif var in 'dPIwspd':
			       varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS]	
			    elif var in 'dPIslp':
			       varRegress.x[count1,count2] = nanmean(bt.PIslp[it-lT:it+1,iS])-bt.StormMslp[it-lT,iS]	
			    elif var in TimeDepends:
                               varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,iS])
                            elif var in 'landmaskMean':
                               varRegress.x[count1,count2] = getattr(bt,var)[it,iS]-getattr(bt,var)[it-lT,iS]
                            elif var in 'StormMwspd':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])
                            elif var in 'StormMwspd2':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**2
                            elif var in 'StormMwspd3':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**3
                            elif var in 'MPI':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])
                            elif var in 'MPI2':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])**2
                            elif var in 'dPIwspd2':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**2
                            elif var in 'dPIwspd3':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**3
                            elif 'dVdt2' in var:
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**2 
                            elif 'dVdt3' in var:
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**3 
                            elif var in 'pre_V0':
                               if it-lT >= 2:
                                  varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it-lT-2,iS]
                               else:
                                  varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it,iS]
                            elif var in 'pre_V02':
                               if it-lT >= 2:
                                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT-2,iS])**2
                               else:
                                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it,iS])**2
                            else:
                               varRegress.x[count1,count2] = getattr(bt,var)[it-lT,iS]

                            count2 += 1
                        count1 += 1 
 
	      iS += 1
              if iS > bt.StormYear.shape[0]-1:break
	print varRegress1.x[0,:]

        a = 1
	for iv in range(0,n2,1):
            a = a*varRegress.x[:,iv] 	
	a = a*varRegress.y

	X = sm.add_constant(varRegress.x[a==a,:],prepend = 'True')
	Y = varRegress.y[a==a]
	X,meanX,stdX,Y,meanY,stdY = normalizeTransform(X,Y) 
	results = sm.OLS(Y,X).fit()
	return results, meanX,meanY,stdX,stdY

def calMultiRegression_coef_mean_waterland(bt,leadingTime,ty1,ty2,predictors,fstType):
	"""
        calculate mean coefficients for multiple regression on water and land
        bt: best track data
        ty1: year1
        ty2: year2
        predictors: predictors
        fstType: fst variable type
	"""
	import copy
	
	lT = np.int(leadingTime/6)
        n1 = bt.StormMslp.shape[0]*bt.StormMslp.shape[1]
        n2 = len(predictors)
        varRegress = Vregression1(n1,n2)
        varRegress1 = Vregression1(n1,n2)
        TimeDepends = ['dThetaEMean','T200Mean','rhMean','rh500_300Mean','div200Mean','T100Mean','dThetaEsMean'] 


	iS = bt.StormYear.tolist().index(ty1) 
        count1 = 0
	while True:
	      if ( (int(bt.StormYear[iS]) >= ty1) & (int(bt.StormYear[iS]) < ty2) ):
		 for it in range(lT,bt.StormLon[:,iS].shape[0],1):
                     #### if the starting or destination point has more than 50% land over 300km radius
                     if((bt.StormLon[it,iS]==bt.StormLon[it,iS])&((bt.landmaskMean[it,iS]<= -0.5)&(bt.landmaskMean[it-lT,iS]<= -0.5))):
                        if (fstType == 'Vmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS] 	
                        elif (fstType == 'dVmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
			count2 = 0
                        var = 'landmaskMean'
                        dldMask = 1.0#getattr(bt,var)[it,iS]-getattr(bt,var)[it-lT,iS]
                        for var in predictors:
                            if var == 'SHRD':
                               varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))
			    elif var == 'dPIwspd':
			       varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS]	
			    elif var in TimeDepends:
                               varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,iS])
                            elif var == 'landmaskMean':
                               varRegress.x[count1,count2] = getattr(bt,var)[it,iS]-getattr(bt,var)[it-lT,iS]
                            elif var == 'StormMwspd':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])
                            elif var == 'StormMwspdL':
                               varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it-lT,iS]*dldMask
                            elif var == 'dPIwspd2':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**2
                            elif var == 'dPIwspd3':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**3
                            elif var == 'dVdtL':
                               varRegress.x[count1,count2] = getattr(bt,'dVdt')[it-lT,iS]*dldMask 
                            elif var == 'dPIwspdL':
			       varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])*dldMask	
                            elif var == 'SHRDL':
                               varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))*dldMask
                            elif var == 'rhMeanL':
                               varRegress.x[count1,count2] = nanmean(getattr(bt,'rhMean')[it-lT:it+1,iS])*dldMask
                            elif var == 'dPIwspd2L':
                               varRegress.x[count1,count2] = dldMask*(nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**2
                            elif var == 'dPIwspd3L':
                               varRegress.x[count1,count2] = dldMask*(nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**3
                            elif var == 'dVdt2L':
                               varRegress.x[count1,count2] = dldMask*(getattr(bt,'dVdt')[it-lT,iS])**2 
                            elif var == 'dVdt2':
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**2 
                            elif var == 'trSpeedL':
                               varRegress.x[count1,count2] = (getattr(bt,'trSpeed')[it-lT,iS])*dldMask 
                            else:
                               varRegress.x[count1,count2] = getattr(bt,var)[it-lT,iS]
                            count2 += 1
                        count1 += 1 
 
	      iS += 1
              if iS > bt.StormYear.shape[0]-1:break
        a = 1
	for iv in range(0,n2,1):
            a = a*varRegress.x[:,iv] 	
	a = a*varRegress.y

	X = sm.add_constant(varRegress.x[a==a,:],prepend = 'True')
	Y = varRegress.y[a==a]
	X,meanX,stdX,Y,meanY,stdY = normalizeTransform(X,Y) 
	results = sm.OLS(Y,X).fit()
        return results,meanX,meanY,stdX,stdY

def calMultiRegression_coef_mean_water(bt,leadingTime,ty1,ty2,predictors,fstType):
	"""
        calculate mean coefficients for multiple regression for water
        bt: best track data
        ty1: year1
        ty2: year2
        predictors: predictors
        fstType: fst variable type
	"""
	import copy
	
	lT = np.int(leadingTime/6)
        n1 = bt.StormMwspd.shape[0]*bt.StormMwspd.shape[1]
        n2 = len(predictors)
        varRegress = Vregression1(n1,n2)
        varRegress1 = Vregression1(n1,n2)
        TimeDepends = ['dThetaEMean','T200Mean','rhMean','rh500_300Mean','div200Mean','T100Mean','dThetaEsMean'] 


	iS = bt.StormYear.tolist().index(ty1) 
        count1 = 0
	while True:
	      if ( (int(bt.StormYear[iS]) >= ty1) & (int(bt.StormYear[iS]) < ty2) ):
		 for it in range(lT,bt.StormLon[:,iS].shape[0],1):
                     #### if the starting or destination point has more than 50% land over 300km radius
                     if((bt.StormLon[it,iS]==bt.StormLon[it,iS])&((bt.landmaskMean[it,iS]<= -0.5)&(bt.landmaskMean[it-lT,iS]<= -0.5))):
                        if (fstType == 'Vmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS] 	
                        elif (fstType == 'dVmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
			count2 = 0
                        for var in predictors:
                            if var in 'SHRD':
                               varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))
			    elif var in 'SHRD_I':
                               varRegress.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
			    elif var in 'dPIwspd':
			       varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS]	
			    elif var in 'dPIslp':
			       varRegress.x[count1,count2] = nanmean(bt.PIslp[it-lT:it+1,iS])-bt.StormMslp[it-lT,iS]	
			    elif var in TimeDepends:
                               varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,iS])
                            elif var in 'landmaskMean':
                               varRegress.x[count1,count2] = getattr(bt,var)[it,iS]-getattr(bt,var)[it-lT,iS]
                            elif var in 'StormMwspd':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])
                            elif var in 'StormMwspd2':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**2
                            elif var in 'StormMwspd3':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**3
                            elif var in 'MPI':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])
                            elif var in 'MPI2':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])**2
                            elif var in 'dPIwspd2':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**2
                            elif var in 'dPIwspd3':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**3
                            elif 'dVdt2' in var:
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**2 
                            elif 'dVdt3' in var:
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**3 
                            elif var in 'pre_V0':
                               if it-lT >= 2:
                                  varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it-lT-2,iS]
                               else:
                                  varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it,iS]
                            elif var in 'pre_V02':
                              if it-lT >= 2:
                                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT-2,iS])**2
                              else:
                                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it,iS])**2
                            else:
                               varRegress.x[count1,count2] = getattr(bt,var)[it-lT,iS]
                            count2 += 1
                        count1 += 1 
 
	      iS += 1
              if iS > bt.StormYear.shape[0]-1:break
	print varRegress1.x[0,:]

        a = 1
	for iv in range(0,n2,1):
            a = a*varRegress.x[:,iv] 	
	a = a*varRegress.y

	X = sm.add_constant(varRegress.x[a==a,:],prepend = 'True')
	Y = varRegress.y[a==a]
	X,meanX,stdX,Y,meanY,stdY = normalizeTransform(X,Y) 
	results = sm.OLS(Y,X).fit()
	return results, meanX,meanY,stdX,stdY

def calPrediction_bt(bt,leadingTime,py1,py2,predictors,pError,countE,coeffFile,meanFile):
        """
        returns error in prediction 
        bt: best track data
        py1: prediction year 1
        py2: prediction year2
        predictors: predictors
        pError: prediction error
        countE: error count
        coeffFile: regression coefficients
        meanFile: file with means
        """
        import copy

        print meanFile
        lT = np.int(leadingTime/6)
        n1 = bt.StormMslp.shape[0]*bt.StormMslp.shape[1]
        n2 = len(predictors)
        varRegress = Vregression1(n1,n2)
        varRegress1 = Vregression1(n1,n2)
        TimeDepends = ['dThetaEMean','T200Mean','rhMean','rh500_300Mean','div200Mean','T100Mean','dThetaEsMean'] 

	#### readin Coefficient
        f = open(coeffFile,'r')
        a = f.readlines()
	ll = a[leadingTime/12].rstrip().rsplit()
	f.close()
	f = open(meanFile)
	b = f.readlines()
	lm = b[leadingTime/12].rstrip().rsplit()
	f.close()
	print len(predictors)


	iS = 0
        count1 = 0
	Y1 = copy.copy(varRegress1.y)
	Y0 = copy.copy(varRegress1.y)
	while True:
	      if ( (int(bt.StormYear[iS]) >= py1) & (int(bt.StormYear[iS]) < py2) ):
		 for it in range(lT,bt.StormLon[:,iS].shape[0],1):
                     if bt.StormLon[it,iS]==bt.StormLon[it,iS]:
                        Y0[count1] = bt.StormMwspd[it,iS]
                        Y1[count1] = bt.StormMwspd[it-lT,iS]
                        varRegress1.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
                        count2 = 0
                        for var in predictors:
                            if var in 'SHRD':
                               varRegress1.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))
			    elif var in 'SHRD_I':
                               varRegress1.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
			    elif var in 'dPIwspd':
			       varRegress1.x[count1,count2] = nanmean(bt.PIwspd[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS]	
			    elif var in 'dPIslp':
			       varRegress1.x[count1,count2] = nanmean(bt.PIslp[it-lT:it+1,iS])-bt.StormMslp[it-lT,iS]	
			    elif var in TimeDepends:
                               varRegress1.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,iS])
			    elif var in 'StormMslp':
                               varRegress1.x[count1,count2] = getattr(bt,var)[it-lT,iS]
                            else:
                               varRegress1.x[count1,count2] = getattr(bt,var)[it-lT,iS]
                            count2 += 1
                        count1 += 1 
 
	      iS += 1
              if iS > bt.StormYear.shape[0]-1:break

        a1 = 1
	for iv in range(0,n2,1):
	    a2 = varRegress1.x[:,iv]
            a1 = a1*varRegress1.x[:,iv] 	
	a1 = a1*varRegress1.y

        Y1 = Y1[a1==a1]
        Y0 = Y0[a1==a1]
	X2 = sm.add_constant(varRegress1.x[a1==a1,:],prepend = 'True')
	Y2 = varRegress1.y[a1==a1,:]
        for iv in range(1,n2+1,1):
	    X2[:,iv]=(X2[:,iv]-np.float(lm[iv+2]))/np.float(lm[iv+2+len(predictors)])

        Y2 = np.float(ll[1])
        for iv in range(1,n2+1,1):
            Y2 = Y2 + X2[:,iv]*np.float(ll[iv+1])

        Y2 = (Y2*np.float(lm[2])) + np.float(lm[1])
        Y2 = Y2 + Y1

        pError.rmse[countE] = np.sqrt(np.mean((Y2 - Y0)**2.))
	pError.mean[countE] = np.mean(abs(Y2-Y0))
	pError.std[countE] = np.std(Y2-Y0)
	print pError.rmse[countE] 


	return pError
        #return Y2, Y0


def calMultiRegression(bt,sf,of,leadingTime,ty1,ty2,py1,py2,predictors,fstType,pError,sError,oError,bError,countE):
	"""
        calculate multiple regression
        bt: best track data
        ty1: year1
        ty2: year2
        predictors: predictors
        fstType: fst variable type
        pError: prediction error
        sError: standard error? 
        oError: observed error
        bError: 
	"""
	import copy
	
	lT = np.int(leadingTime/6)
        n1 = bt.StormMslp.shape[0]*bt.StormMslp.shape[1]
        n2 = len(predictors)
        varRegress = Vregression1(n1,n2)
        varRegress1 = Vregression1(n1,n2)
        TimeDepends = ['dThetaMean','T200Mean','rhMean','rh500_300Mean','div200Mean'] 
 	#,'SHRD','dThetaEMean','T200Mean','rhMean','rh500_300Mean']


	iS = 0
        count1 = 0
	while True:
	      if ( (int(bt.StormYear[iS]) >= ty1) & (int(bt.StormYear[iS]) < ty2) ):
		 for it in range(lT,bt.StormLon[:,iS].shape[0],1):
                     if bt.StormLon[it,iS]==bt.StormLon[it,iS]:
                        if (fstType == 'Vmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS] 	
                        elif (fstType == 'dVmax'):
                           varRegress.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
			count2 = 0
                        for var in predictors:
                            if var in 'SHRD':
                               varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it,iS]**2+bt.VShearMean[it-lT:it,iS]**2))
			    elif var in 'SHRD_I':
                               varRegress.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
			    elif var in 'dPIwspd':
			       varRegress.x[count1,count2] = nanmean(bt.PIwspd[it-lT:it,iS])-bt.StormMwspd[it-lT,iS]	
			    elif var in TimeDepends:
                               varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it,iS])
                            else:
                               varRegress.x[count1,count2] = getattr(bt,var)[it-lT,iS]
                            count2 += 1
                        count1 += 1 
 
	      iS += 1
              if iS > bt.StormYear.shape[0]-1:break

        a = 1
	for iv in range(0,n2,1):
            a = a*varRegress.x[:,iv] 	
	a = a*varRegress.y

	X = sm.add_constant(varRegress.x[a==a,:],prepend = 'True')
	Y = varRegress.y[a==a]
	X,meanX,stdX,Y,meanY,stdY = normalizeTransform(X,Y) 
	results = sm.OLS(Y,X).fit()


	iS = 0
        count1 = 0
	Y1 = copy.copy(varRegress1.y)
	Y0 = copy.copy(varRegress1.y)
	while True:
	      if ( (int(bt.StormYear[iS]) >= py1) & (int(bt.StormYear[iS]) < py2) ):
		 for it in range(lT,bt.StormLon[:,iS].shape[0],1):
                     if bt.StormLon[it,iS]==bt.StormLon[it,iS]:
                        Y0[count1] = bt.StormMwspd[it,iS]
                        Y1[count1] = bt.StormMwspd[it-lT,iS]
                        varRegress1.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
                        count2 = 0
                        for var in predictors:
                            if var in 'SHRD':
                               varRegress1.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it,iS]**2+bt.VShearMean[it-lT:it,iS]**2))
			    elif var in 'SHRD_I':
                               varRegress1.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
			    elif var in 'dPIwspd':
			       varRegress1.x[count1,count2] = nanmean(bt.PIwspd[it-lT:it,iS])-bt.StormMwspd[it-lT,iS]	
			    elif var in TimeDepends:
                               varRegress1.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it,iS])
                            else:
                               varRegress1.x[count1,count2] = getattr(bt,var)[it-lT,iS]
                            count2 += 1
                        count1 += 1 
 
	      iS += 1
              if iS > bt.StormYear.shape[0]-1:break

        a1 = 1
	for iv in range(0,n2,1):
            a1 = a1*varRegress1.x[:,iv] 	
	a1 = a1*varRegress1.y

        Y1 = Y1[a1==a1]
        Y0 = Y0[a1==a1]
	X2 = sm.add_constant(varRegress1.x[a1==a1,:],prepend = 'True')
	Y2 = varRegress1.y[a1==a1,:]
        for iv in range(1,n2+1,1):
	    X2[:,iv]=(X2[:,iv]-meanX[iv])/stdX[iv]

        Y2 = results.params[0]
        for iv in range(1,n2+1,1):
            Y2 = Y2 + X2[:,iv]*results.params[iv]

        Y2 = (Y2*stdY) + meanY
        Y2 = Y2 + Y1


	iS = 0
        sfVtrue = np.array([])
        sfVfst = np.array([])
        while True:
              if ( (sf.StormYear[iS] >= py1) & (sf.StormYear[iS] <= py2) ):
                 for it in range(lT,sf.VMAX.shape[1],1):
                     sfVtrue = np.append(sfVtrue,sf.VMAX[iS,it,0]);
                     sfVfst  = np.append(sfVfst,sf.VMAX[iS,it-lT,lT]);
              iS += 1
              if iS > sf.StormYear.shape[0]-1:break
        sfVfst[sfVfst>=9999. ]=float('Nan')
        sfVfst[sfVfst==0. ]=float('Nan')
        sfVtrue[sfVtrue>=9999. ]=float('Nan')
        sfVtrue[sfVtrue==0. ]=float('Nan')
        sfVfst1=sfVfst[(sfVfst==sfVfst)&(sfVtrue==sfVtrue)]
        sfVtrue1=sfVtrue[(sfVfst==sfVfst)&(sfVtrue==sfVtrue)]

	iS = 0
        ofVtrue = np.array([])
        ofVfst = np.array([])
        while True:
              if ( (of.StormYear[iS] >= py1) & (of.StormYear[iS] <= py2) ):
                 for it in range(lT,of.VMAX.shape[1],1):
                     ofVtrue = np.append(ofVtrue,of.VMAX[iS,it,0]);
                     ofVfst  = np.append(ofVfst,of.VMAX[iS,it-lT,lT]);
              iS += 1
              if iS > of.StormYear.shape[0]-1:break
        ofVfst[ofVfst>=9999. ]=float('Nan')
        ofVfst[ofVfst==0. ]=float('Nan')
        ofVtrue[ofVtrue>=9999. ]=float('Nan')
        ofVtrue[ofVtrue==0. ]=float('Nan')
        ofVfst1=ofVfst[(ofVfst==ofVfst)&(ofVtrue==ofVtrue)]
        ofVtrue1=ofVtrue[(ofVfst==ofVfst)&(ofVtrue==ofVtrue)]



        fig = plt.figure(figsize=(3.25,3.25))
        ax = fig.add_subplot(111)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        plt.ylim([0,200])
        plt.xlim([0,200])
        plt.plot(sfVtrue1,sfVfst1,'k.',label='SHIPS '+str(py1)+'-'+str(py2)+' ('\
                +str(sfVtrue1.shape[0])+')',markersize=5)
        plt.plot(ofVtrue1,ofVfst1,'k.',label='OFCL '+str(py1)+'-'+str(py2)+' ('\
                +str(ofVtrue1.shape[0])+')',markersize=4,color='gray',markeredgecolor='gray')
        plt.plot(Y0,Y2,'r.',label='Predicted'+str(py1)+'-'+str(py2)+' ('\
                +str(Y2.shape[0])+')',markersize=3,markeredgecolor = 'r')
        plt.plot([0,200],[0,200],color='gray')
        fontproperty=matplotlib.font_manager.FontProperties()
        fontproperty.set_size('small')
        plt.legend(loc='upper left', prop={'size':6})
        plt.xlabel('Vmax-bt [kt]',fontsize=8)
        plt.ylabel('Vmax-fst [kt]',fontsize=8)
        plt.title('Lead time: '+int2str(leadingTime,3)+' hr  ',fontsize=8)
        plt.savefig('scatter_'+fstType+'_'+int2str(leadingTime,3)+'hr.png',dpi=300)
        plt.clf()

        pError.rmse[countE] = np.sqrt(np.mean((Y2 - Y0)**2.))
        bError.rmse[countE] = np.sqrt(np.mean((Y1 - Y0)**2.))
        sError.rmse[countE] = np.sqrt(np.mean((sfVfst1-sfVtrue1)**2.))
        oError.rmse[countE] = np.sqrt(np.mean((ofVfst1-ofVtrue1)**2.))
	pError.mean[countE] = np.mean(abs(Y2-Y0))
	bError.mean[countE] = np.mean(abs(Y1-Y0))
	sError.mean[countE] = np.mean(abs(sfVfst1-sfVtrue1))
	oError.mean[countE] = np.mean(abs(ofVfst1-ofVtrue1))
#	pError.mean[countE] = np.mean((Y1-Y2))
#	sError.mean[countE] = np.mean((sfVfst1-sfVtrue1))
#	oError.mean[countE] = np.mean((ofVfst1-ofVtrue1))
	pError.std[countE] = np.std(Y2-Y0)
	bError.std[countE] = np.std(Y1-Y0)
	sError.std[countE] = np.std(sfVfst1-sfVtrue1)
	oError.std[countE] = np.std(ofVfst1-ofVtrue1)


        return results,pError,sError,oError,bError

def normalizeTransform(X,Y):
        """
        Find mean and standard deviation of X and Y
        """
        meanX = []
        stdX = []
        meanX.append(0.0)
        stdX.append(0.0)
        meanY = np.mean(Y)
        stdY = np.std(Y)
        Y = (Y - meanY)/stdY
        for ix in range(1,X.shape[1],1):
            meanX.append(np.mean(X[:,ix]))
            stdX.append(np.std(X[:,ix]))
            X[:,ix] = (X[:,ix] - meanX[ix])/stdX[ix]

        return X,meanX,stdX,Y,meanY,stdY

def singleStorm(id,year,name,of,sf,bt,coeffFile,meanFile,predictors,TimeDepends):
	"""
        plot single storm
        id: storm id
        year: year of single storm
        name: name of storm
        of: observed file
        sf: storm file
        bt: best track data
        coeffFile: file with coefficients
        meanFile: file with means
        predictors: predictors
	"""
	fig = plt.figure(figsize=(6.5,3.25))
	ax = fig.add_subplot(111)
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(6)
	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(6)

	f = open(meanFile)
	b = f.readlines()
	f.close()

	# find index for ships data for the given storm year and ID
	for iS in range(0,sf.StormYear.shape[0],1):
	    if ((sf.StormYear[iS]==year) & (sf.StormId[iS] == id)):
	       index = iS
	# see how many time the fst was issued for the storm
	for iS in range(0,sf.StormTime.shape[1],2):
	    # see if the dat is not empty
	    if sf.StormTime[index,iS]:
	       time1 = sf.fstT[index,iS,:][(sf.fstT[index,iS,:]<9999.0)&(sf.VMAX[index,iS,:]==sf.VMAX[index,iS,:])]
	       v1 = sf.VMAX[index,iS,:][(sf.fstT[index,iS,:]<9999.0)&(sf.VMAX[index,iS,:]==sf.VMAX[index,iS,:])]
	       v1[v1>=300.]=np.float('nan')
	       time2 = np.array([])
	       #plot predicted Vmax for that forecast
	       for iv1 in range(0,v1.shape[0],1):
	           time2=np.append(time2,sf.StormTime[index,iS] + timedelta(hours = time1[iv1]))
	       plt. plot(time2,v1,'k-',color='blue')

	# same thing for official forecast
	for iS in range(0,of.StormYear.shape[0],1):
	    if ((of.StormYear[iS]==year) and (of.StormId[iS] == id)):
	       index = iS

	for iS in range(0,of.StormTime.shape[1],2):
	    if of.StormTime[index,iS]:
	       time1 = of.fstT[index,iS,:][(of.fstT[index,iS,:]<9999.0)&(of.VMAX[index,iS,:]==of.VMAX[index,iS,:])]
	       if time1.any():
	          v1 = of.VMAX[index,iS,:][(of.fstT[index,iS,:]<9999.0)&(of.VMAX[index,iS,:]==of.VMAX[index,iS,:])]
		  v1[v1>=300.]=np.float('nan')
	          time2 = np.array([])
	          for iv1 in range(0,v1.shape[0],1):
	              time2=np.append(time2,of.StormTime[index,iS] + timedelta(hours = time1[iv1]))
	          plt.plot(time2,v1,'k-',color='gray')
	          del time2, v1, time1

	# find the index on the best-track data
	index = bt.StormYear.tolist().index(year)
	index = index + bt.StormId[index:].tolist().index(id)
	# for each forecast, we issue 120hr forecast every 12 hours
	fstperiod = range(12,132,12)
	# now we need to find the mean and std for the predictors and dependent variable
	n1 = bt.StormMwspd[:,index].shape[0]
	n2 = len(predictors)
	meanX = np.empty([len(fstperiod),n2],dtype=float)
	meanY = np.empty([len(fstperiod)],dtype=float)
	stdX  = np.empty([len(fstperiod),n2],dtype=float)
	stdY  = np.empty([len(fstperiod)],dtype=float)
	count1 = 0
	for ih in fstperiod:
	    lm = b[ih/12].rstrip().rsplit()
    	    meanX[count1,:] = lm[3:9]
            meanY[count1]  = lm[1]
            stdX[count1,:] = lm[9::] 
            stdY[count1]   = lm[2]
	    count1 += 1
	lT = ih/6
	for it in range(0,bt.StormMwspd.shape[0],2):
	    if (((it+lT)<=bt.StormMwspd.shape[0]) and (bt.StormMwspd[it,index]==bt.StormMwspd[it,index])):
	       h1,v1 = getPrediction(bt,meanX,meanY,stdX,stdY,it, index, fstperiod,coeffFile,predictors,TimeDepends)	
	       plt.plot(h1,v1,'r-')	


	plt.plot(bt.Time[:,index],bt.StormMwspd[:,index],'k-')
	days    = DayLocator(interval=3)
	hours   = HourLocator(range(0,24,6))
	daysFmt = DateFormatter('%b %d')
	ax.xaxis.set_major_locator(days)
	ax.xaxis.set_major_formatter(daysFmt)

	fontproperty=matplotlib.font_manager.FontProperties()
	fontproperty.set_size('small')
	fig.subplots_adjust(left=0.15, right = 0.9, bottom=0.15)
	plt.xlabel('Date',fontsize=8)
	plt.ylabel('Vmax[kt]',fontsize=8)
	plt.title('Hurricane '+name+' ('+str(year)+'): blue-SHIPS, gray-OFCL, red',fontsize=8)
	plt.savefig(str(year)+name+'.png',dpi=300)



	return


def getVariables(bt,iSS,lT,predictors):
	"""
        This function calculates variable values.
        bt: best track data
        iSS: index
        lT: index
        predictors: predictors
	"""
        TimeDepends = ['dThetaEMean','T200Mean','rhMean','rh500_300Mean','div200Mean','T100Mean','dThetaEsMean']
        n1 = iSS.shape[0]*bt.StormMslp.shape[0]
	n2 = len(predictors)
        print n1, n2
	varRegress = Vregression1(n1,n2)
        count1 = 0
        for index in iSS:
	  for it in range(0,bt.StormMwspd.shape[0],2):
	    if (it+lT) < bt.StormMwspd.shape[0]:
	       if (bt.StormLon[it,index] == bt.StormLon[it,index]) and (bt.StormLon[it+lT,index] == bt.StormLon[it+lT,index]):
		  count2 = 0
                  varRegress.y[count1]=bt.StormMwspd[it+lT,index] - bt.StormMwspd[it,index]
		  for var in predictors:
                            if var in 'SHRD':
                               varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,index]**2+bt.VShearMean[it-lT:it+1,index]**2))
                            elif var in 'SHRD_I':
                               varRegress.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,index]**2+bt.VShearMean[it-lT,index]**2)
                            elif var in 'dPIwspd':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,index])-bt.StormMwspd[it-lT,index]
                            elif var in 'dPIslp':
                               varRegress.x[count1,count2] = nanmean(bt.PIslp[it-lT:it+1,index])-bt.StormMslp[it-lT,index]
                            elif var in TimeDepends:
                               varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,index])
                            elif var in 'landmaskMean':
                               varRegress.x[count1,count2] = getattr(bt,var)[it,index]-getattr(bt,var)[it-lT,index]
                            elif var in 'StormMwspd':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,index])
                            elif var in 'StormMwspd2':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,index])**2
                            elif var in 'StormMwspd3':
                               varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,index])**3
                            elif var in 'MPI':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,index])
                            elif var in 'MPI2':
                               varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,index])**2
                            elif var in 'dPIwspd2':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,index])-bt.StormMwspd[it-lT,index])**2
                            elif var in 'dPIwspd3':
                               varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,index])-bt.StormMwspd[it-lT,index])**3
                            elif 'dVdt2' in var:
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,index])**2
                            elif 'dVdt3' in var:
                               varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,index])**3
                            elif var in 'pre_V0':
                               if it-lT >= 2:
                                  varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it-lT-2,index]
                               else:
                                  varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it,index]
                            elif var in 'pre_V02':
                              if it-lT >= 2:
                                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT-2,index])**2
                              else:
                                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it,index])**2
                            else:
                               varRegress.x[count1,count2] = getattr(bt,var)[it-lT,index]
                            count2 += 1
                  count1 += 1

	a = 1
	for iv in range(0,n2,1):
	    a = a*varRegress.x[:,iv]
        
        print a[a==a].shape
	a = a*varRegress.y
	X1 = sm.add_constant(varRegress.x[a==a,:],prepend = 'True')
	Y1 = varRegress.y[a==a]

	return X1, Y1

def getPrediction(bt,meanX,meanY,stdX,stdY,it,index,fstperiod,coeffFile,predictors,TimeDepends):
        '''
        This function returns predictions of intensities and times in two arrays.
        bt: best track data
        meanX: mean X
        meanY: mean Y
        stdX: standard deviation in X
        stdY: standard deviation in y
        it: time in iteration
        coeffFile: file with coefficients
        '''
	f = open(coeffFile,'r')
	a = f.readlines()
	v1 = []
	h1 = []
	v1.append(bt.StormMwspd[it,index])
	h1.append(bt.Time[it,index])
	for ih in fstperiod:
	    h1.append(bt.Time[it,index]+timedelta(hours = ih))
	    ll = a[ih/12].rstrip().rsplit()
	    lT = ih/6
	    dy = np.float(ll[1])
	    count2 = 0
            for var in predictors:
		#print meanX.shape
                if var in 'SHRD':
                  rvar =nanmean(np.sqrt(bt.UShearMean[it:it+lT,index]**2+bt.VShearMean[it:it+lT,index]**2))
                elif var in 'dPIwspd':
                   rvar = nanmean(bt.PIwspd[it:it+lT,index])-bt.StormMwspd[it,index]
                elif var in TimeDepends:
                   rvar = nanmean(getattr(bt,var)[it:it+lT,index])
                else:
                   rvar = getattr(bt,var)[it,index]
                rvar = (rvar-meanX[ih/12-1,count2])/stdX[ih/12-1,count2]
                dy = dy + rvar*np.float(ll[count2+2])
                count2 = count2 + 1
            dy = (dy*stdY[ih/12-1])+meanY[ih/12-1]
            v1.append(bt.StormMwspd[it,index]+dy)

	return h1,v1
def getPrediction_v0input(bt,meanX,meanY,stdX,stdY,it,index,fstperiod,coeffFile,predictors,TimeDepends,v0,dvdt):
        '''
        This function returns predictions of intensities and times in two arrays based on v0 input
        bt: best track data
        meanX: mean X
        meanY: mean Y
        stdX: standard deviation in X
        stdY: standard deviation in y
        it: time in iteration
        coeffFile: file with coefficients
        '''
        f = open(coeffFile,'r')
        a = f.readlines()
        v1 = []
        h1 = []
        v1.append(bt.StormMwspd[it,index])
        h1.append(bt.Time[it,index])
        for ih in fstperiod:
            h1.append(bt.Time[it,index]+timedelta(hours = ih))
            ll = a[ih/12].rstrip().rsplit()
            lT = ih/6
            dy = np.float(ll[1])
            count2 = 0
            for var in predictors:
                #print meanX.shape
                if var in 'SHRD':
                  rvar =nanmean(np.sqrt(bt.UShearMean[it:it+lT,index]**2+bt.VShearMean[it:it+lT,index]**2))
                elif var in 'dPIwspd':
                   rvar = nanmean(bt.PIwspd[it:it+lT,index])-v0
                elif var in TimeDepends:
                   rvar = nanmean(getattr(bt,var)[it:it+lT,index])
                elif var in 'StormMwspd':
                   rvar = v0
                elif var in 'dVdt':
                   rvar = dvdt
                else:
                   rvar = getattr(bt,var)[it,index]
                rvar = (rvar-meanX[ih/12-1,count2])/stdX[ih/12-1,count2]
                dy = dy + rvar*np.float(ll[count2+2])
                count2 = count2 + 1
            dy = (dy*stdY[ih/12-1])+meanY[ih/12-1]
            v1.append(bt.StormMwspd[it,index]+dy)

        return h1,v1

def getPrediction_v0input_dynamic(bt,meanX,meanY,stdX,stdY,it,index,fstperiod,coeffFile,predictors,TimeDepends,v0,dvdt,coeff_pi,bins_pi):
        '''
        This function returns dynamic predictions of intensities and times in two arrays based on v0 input.
        bt: best track data
        meanX: mean X
        meanY: mean Y
        stdX: standard deviation in X
        stdY: standard deviation in y
        it: time in iteration
        coeffFile: file with coefficients
        '''
        f = open(coeffFile,'r')
        a = f.readlines()
        v1 = []
        h1 = []
        v1.append(bt.StormMwspd[it,index])
        h1.append(bt.Time[it,index])
        for ih in fstperiod:
            h1.append(bt.Time[it,index]+timedelta(hours = ih))
            ll = a[ih/12].rstrip().rsplit()
            lT = ih/6
            dy = np.float(ll[1])
            count2 = 0
            for var in predictors:
                #print meanX.shape
                if var in 'SHRD':
                  rvar =nanmean(np.sqrt(bt.UShearMean[it:it+lT,index]**2+bt.VShearMean[it:it+lT,index]**2))
                elif var in 'dPIwspd':
                   rvar = nanmean(bt.PIwspd[it:it+lT,index])-v0
                elif var in TimeDepends:
                   rvar = nanmean(getattr(bt,var)[it:it+lT,index])
                elif var in 'StormMwspd':
                   rvar = v0
                elif var in 'dVdt':
                   rvar = dvdt
                else:
                   rvar = getattr(bt,var)[it,index]
                rvar = (rvar-meanX[ih/12-1,count2])/stdX[ih/12-1,count2]
                if var in 'dPIwspd':
                   i_pi,j_pi = find_range(bins_pi[:,ih/12-1],rvar)
                   if i_pi == -1:
                      i0 = 0
                   elif j_pi == bins_pi.shape[0]:
                      i0 = bins_pi.shape[0]-1
                   else:
                      i0 = i_pi
                   rvar = rvar*coeff_pi[i0,ih/12-1]
                else:
                   rvar = rvar*np.float(ll[count2+2])
                dy = dy + rvar
                count2 = count2 + 1
            dy = (dy*stdY[ih/12-1])+meanY[ih/12-1]
            v1.append(bt.StormMwspd[it,index]+dy)
        return h1,v1
	
def getPrediction_oneTime(bt,meanX,meanY,stdX,stdY,it,index,ih,coeffFile,predictors,TimeDepends):
        '''
        This function returns predictions of intensities and times in two arraysat one time.
        bt: best track data
        meanX: mean X
        meanY: mean Y
        stdX: standard deviation in X
        stdY: standard deviation in y
        it: time in iteration
        coeffFile: file with coefficients
        '''
        f = open(coeffFile,'r')
        a = f.readlines()
        ll = a[ih/12].rstrip().rsplit()
        lT = ih/6
        count2 = 0
        dy = np.float(ll[1])
        for var1 in predictors:
            if var1 in 'SHRD':
               rvar = nanmean(np.sqrt(bt.UShearMean[it:it+lT,index]**2+bt.VShearMean[it:it+lT,index]**2))
            elif var1 in 'SHRD_I':
               rvar = np.sqrt(bt.UShearMean[it,index]**2+bt.VShearMean[it,index]**2)
            elif var1 in 'dPIwspd':
               rvar = nanmean(bt.PIwspd[it:it+lT,index])-bt.StormMwspd[it,index]
            elif var1 in TimeDepends:
               rvar = nanmean(getattr(bt,var1)[it:it+lT,index])
            else:
               rvar = getattr(bt,var1)[it,index]
            rvar = (rvar - meanX[count2+1])/stdX[count2+1]
            dy = dy + rvar*np.float(ll[count2+2])
            count2 += 1
	dy = dy*stdY+meanY
        return dy 

def getPrediction_v0input(bt,meanX,meanY,stdX,stdY,it,index,fstperiod,coeffFile,predictors,TimeDepends,v0,dvdt):
        '''
        gets prediction 
        bt: best track data 
        meanX: mean X 
        meanY: mean Y 
        stdX: standard deviation in x
        stdY: standard deviation in y
        it: time 
        index: index
        coeffFile: coefficients file 
        predictors: predictors 
        '''
        f = open(coeffFile,'r')
        a = f.readlines()
        v1 = []
        h1 = []
        v1.append(v0)
        h1.append(bt.Time[it,index])
        for ih in fstperiod:
            h1.append(bt.Time[it,index]+timedelta(hours = ih))
            ll = a[ih/12].rstrip().rsplit()
            lT = ih/6
            dy = np.float(ll[1])
            count2 = 0
            for var in predictors:
                #print meanX.shape
                if var in 'SHRD':
                  rvar =nanmean(np.sqrt(bt.UShearMean[it:it+lT+1,index]**2+bt.VShearMean[it:it+lT+1,index]**2))
                elif var in 'dPIwspd':
                   rvar = nanmean(bt.PIwspdMean[it:it+lT+1,index])-v0
                elif var in TimeDepends:
                   rvar = nanmean(getattr(bt,var)[it:it+lT+1,index])
                elif var in 'StormMwspd':
                   rvar = v0
                elif var in 'dVdt':
                   rvar = dvdt
                elif var in 'landmaskMean':
                   rvar = getattr(bt,var)[it+lT+1,index]-getattr(bt,var)[it,index]
                elif var in 'StormMwspd2':
                   rvar = (getattr(bt,'StormMwspd')[it,index])**2
                elif var in 'StormMwspd3':
                   rvar = (getattr(bt,'StormMwspd')[it,index])**3
                elif var in 'MPI':
                   rvar = nanmean(bt.PIwspdMean[it:it+lT+1,index])
                elif var in 'MPI2':
                   rvar = nanmean(bt.PIwspdMean[it:it+lT+1,index])**2
                elif var in 'dPIwspd2':
                   rvar = (nanmean(bt.PIwspdMean[it:it+lT+1,index])-v0)**2
                elif var in 'dPIwspd3':
                   rvar = (1.5*nanmean(bt.PIwspdMean[it:it+lT+1,index])-v0)**3
                elif 'dVdt2' in var:
                   rvar = dvdt**2
                elif 'dVdt3' in var:
                   rvar = dvdt**3
                else:
                   rvar = getattr(bt,var)[it,index]
                #print var, rvar
                rvar = (rvar-meanX[ih/12-1,count2])/stdX[ih/12-1,count2]
                dy = dy + rvar*np.float(ll[count2+2])
                #print count2,var, rvar*np.float(ll[count2+2])
                count2 = count2 + 1
            dy = (dy*stdY[ih/12-1])+meanY[ih/12-1]
            v1.append(v0+dy)

        return h1,v1
def getPrediction_v0input_result(bt,meanX,meanY,stdX,stdY,it,index,fstperiod,result,predictors,TimeDepends,v0,dvdt):
        '''
        gets prediction 
        bt: best track data 
        meanX: mean X 
        meanY: mean Y 
        stdX: standard deviation in x
        stdY: standard deviation in y
        it: time 
        index: index
        coeffFile: coefficients file 
        predictors: predictors 
        '''
        v1 = []
        h1 = []
        v1.append(v0)
        h1.append(bt.Time[it,index])
        for ih in fstperiod:
            h1.append(bt.Time[it,index]+timedelta(hours = ih))
            lT = ih/6
            #dyi = ma.getdata(result['params'][:])
            dy = result[0]
            count2 = 0
            for var in predictors:
                #print meanX.shape
                if var in 'SHRD':
                  rvar =nanmean(np.sqrt(bt.UShearMean[it:it+lT+1,index]**2+bt.VShearMean[it:it+lT+1,index]**2))
                elif var in 'dPIwspd':
                   rvar = nanmean(bt.PIwspdMean[it:it+lT+1,index])-v0
                elif var in TimeDepends:
                   rvar = nanmean(getattr(bt,var)[it:it+lT+1,index])
                elif var in 'StormMwspd':
                   rvar = v0
                elif var in 'dVdt':
                   rvar = dvdt
                elif var in 'landmaskMean':
                   rvar = getattr(bt,var)[it+lT+1,index]-getattr(bt,var)[it,index]
                elif var in 'StormMwspd2':
                   rvar = (getattr(bt,'StormMwspd')[it,index])**2
                elif var in 'StormMwspd3':
                   rvar = (getattr(bt,'StormMwspd')[it,index])**3
                elif var in 'MPI':
                   rvar = nanmean(bt.PIwspdMean[it:it+lT+1,index])
                elif var in 'MPI2':
                   rvar = nanmean(bt.PIwspdMean[it:it+lT+1,index])**2
                elif var in 'dPIwspd2':
                   rvar = (nanmean(bt.PIwspdMean[it:it+lT+1,index])-v0)**2
                elif var in 'dPIwspd3':
                   rvar = (nanmean(bt.PIwspdMean[it:it+lT+1,index])-v0)**3
                elif 'dVdt2' in var:
                   rvar = dvdt**2
                elif 'dVdt3' in var:
                   rvar = dvdt**3
                else:
                   rvar = getattr(bt,var)[it,index]
                #print var, rvar
                rvar = (rvar-meanX[count2+1])/stdX[count2+1]
                dy = dy + rvar*result[count2+1]
                count2 = count2 + 1
            dy = (dy*stdY)+meanY
            v1.append(v0+dy)
        return h1,v1

