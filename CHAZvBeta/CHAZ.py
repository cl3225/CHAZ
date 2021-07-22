#!/usr/bin/env python
import numpy as np
import pandas as pd
import subprocess
import dask.array as da
import os
import src.module_riskModel as mrisk
import src.module_GenBamPred as GBP
import Namelist as gv
import src.caldetsto_track_lysis as CHAZ_detsto
import src.calWindCov as calWindCov
import src.preprocess as preprocess
import src.calA as calA

print gv.Model, gv.ENS, gv.TCGIinput

#######################
### Pre-Processes #####
#######################
if gv.runPreprocess: 
       if gv.calWind:
          calWindCov.run_windCov()
       if gv.calpreProcess:
          preprocess.run_preProcesses()
       if gv.calA:
          calA.run_calA()

#######################
### Run CHAZ       ####
#######################
if gv.runCHAZ:
        ### get seeding ratio in this experiment
        if gv.TCGIinput != 'random':
           print 'get Seeding ratio for', gv.Model, gv.ENS
           ipath = gv.pre_path
           ratio = GBP.get_seeding_ratio(ipath,1981,2005,'HIST')
           print 'The seeding ratio is',ratio

        ### running genesis,track,predictors,& intensity
        ### output yearly 
        for ichaz in range(gv.CHAZ_ENS_0,gv.CHAZ_ENS):
           for iy in range(gv.Year1,gv.Year2+1):
	    ### EXP doesn't matter anymore but we have not remove it in the rest of codes ###
            global EXP,exp,fpath
            EXP='HIST' if iy<=2005 else 'RCP85'
            exp='hist' if iy>2005 else 'rcp85'
            fpath = gv.pre_path
            ### genesis 
            if gv.calGen:
               if gv.TCGIinput == 'random':
                  print iy,'random seeding has not yet tested'
                  #climInitDate, climInitLon, climInitLat = GBP.randomSeeding(iy,gv.seedN)
               else:
                  print iy, 'TCGI'
                  climInitDate, climInitLon, climInitLat = GBP.getSeeding(fpath,iy,EXP,ratio)
            if gv.calBam:
               print iy, 'Bam'
               fst = GBP.getBam(climInitDate,climInitLon,climInitLat,iy,ichaz,EXP)
               print iy, 'calculate predictors'
               GBP.calPredictors(fst,iy,ichaz,EXP)
               del fst
            if gv.calInt:
               print iy, 'calculate Intensitys'
               CHAZ_detsto.calIntensity(EXP,iy,ichaz)

        dir_name = gv.ipath
        t = os.listdir(dir_name)

        for item in t:
            if item.endswith(".pik"):
                os.remove(os.path.join(dir_name, item))              
