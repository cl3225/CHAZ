HAZ (Columbia Tropical Cyclone Hazard Model)

## Table of Contents
I. [Overview](https://github.com/cl3225/CHAZ.github.io#i-overview)

II. [Getting Started With CHAZ](https://github.com/cl3225/CHAZ.github.io#ii-getting-started-with-chaz)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1) [Getting the code](https://github.com/cl3225/CHAZ.github.io#1-getting-the-code)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2) [System Dependencies](https://github.com/cl3225/CHAZ.github.io#2-system-dependencies)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3) [Operating System](https://github.com/cl3225/CHAZ.github.io#3-operating-system)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4) [Setting Up Python Environment](https://github.com/cl3225/CHAZ.github.io#4-setting-up-python-environment)

III. [Basic CHAZ Structure](https://github.com/cl3225/CHAZ.github.io#iii-basic-chaz-structure)

IV. [CHAZ-pre](https://github.com/cl3225/CHAZ.github.io#iv-chaz-pre)

V. [Checking Initial conditions](https://github.com/cl3225/CHAZ.github.io#v-checking-initial-conditions)

VI. [Running CHAZ (Downscaling)](https://github.com/cl3225/CHAZ.github.io#vi-running-chaz-downscaling)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1) [Changing Global Variables in `Namelist.py`](https://github.com/cl3225/CHAZ.github.io#1-changing-global-variables-in-namelistpy)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2) [CHAZ Downscaling](https://github.com/cl3225/CHAZ.github.io#2-chaz-downscaling)

VII. [Output of CHAZ](https://github.com/cl3225/CHAZ.github.io#vii-output-of-chaz)

VIII. [Disclaimer](https://github.com/cl3225/CHAZ.github.io#disclaimer)

IX. [License](https://github.com/cl3225/CHAZ.github.io#license)
		
## I. Overview 


![flow_chart](https://user-images.githubusercontent.com/46905677/126520861-3761c297-4651-43c2-b275-863dca35ad52.png)


The Columbia HAZard model (CHAZ) is a statistical-dynamical downscaling model for estimating tropical cyclone hazard. The CHAZ model consists of three primary components for describing tropical cyclone activity from genesis to lysis; these components are genesis, track, and intensity modules. The genesis module uses Tropical Cyclone Genesis Index (TCGI, developed by Tippett et al, 2011) to estimate the seeding rate of the storm precursors. The seeds are then passed to a  Beta-Advection Model (BAM) that moves the storm forward with giving environmental sterring flow. The intensity module then evolves storms beyond genesis using an Autoregressive Model that involves a deterministic and a stochastic forcing elements. The underlying science of the CHAZ model can be found at [Lee et al. 2018](https://doi.org/10.1002/2017MS001186)).

Generally speaking, there are two primary steps for running CHAZ: preprocessing and downscaling.  Preprocessing includes collecting global model data, calculating PI, TCGI, wind covariance matrix, putting the data into a standard format for downscaling calculation, etc. Preprocessing codes are model-dependent, meaning that you will need to modify them when switch global models. Codes for conducting downscaling is insensitive to what global model you are using. 

## II. Getting Started with CHAZ

### 1) Getting the code

TBD 

### 2) System dependencies 

The CHAZ model was developed in python2 and has only been running on linux and Mac OSX machines. Here we provide our experience. However, this does not mean CHAZ can only be run on a linux system. We welcome users' contributions in terms of operating system. We also appreciate any comments, bug-discovery, and improvement of the CHAZ model.

### 3) Operating System

* Linux/ Mac OSX


### 4) Setting Up Python Environment

1) Install Anaconda 3 or Python 2 
2) Setting up virtual environment

	2a. If [Anaconda Python Distribution](https://www.anaconda.com/) is Installed, run the following commands : 

       conda create --name py27 python=2.7 anaconda

       conda activate py27
       
	2b. If Python directly installed, run the following commands: 

       virtualenv -p /usr/bin/python2.7 venv

       source venv/bin/activate

3) pip install the following code:

       pip install numpy==1.15.4

       pip install pandas==0.24.1

       pip install scipy==1.2.0

       pip install netcdf4==1.4.2

       pip install matplotlib==2.2.3

       pip install xarray==0.11.3

       pip install statsmodels==0.9.0

       pip install cftime==1.0.3.4

       pip install netcdftime==1.0.0a2

       pip install dask==1.1.1
       
       pip install toolz==0.9.0
       


## III. Basic CHAZ structure

In CHAZ, Namelist.py defines all global variables such as the source of the global model forcing, the ensemble members of that global model, the version of genesis module, numbers of track and intensity ensemble members, and whether you are doing CHAZ-pre or CHAZ downscaling. In theory, Namelist.py is the only file you need to modify. CHAZ.py reads in Namelist.py and is the script that calls for all the modules/subroutines. Below we describe input variables that need to be modified when conducting CHAZ-pre and CHAZ downscaling.  

`Namelist.py`: python file containing global variables modified for CHAZ preprocessing and CHAZ downscaling

`CHAZ.py`: python file that controls preprocessing and downscaling

`getMeanStd.py`: used to calculate data in `coefficient_meanstd.nc` if user does not have file (must execute separately from CHAZ)




`input`: the directory containing default input data necessary for running CHAZ. These data are generated during the CHAZ development stage and are used in Lee et al. 2018

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `bt_*.nc`: best track basin-wide inputs for CHAZ
       
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `bt_global_predictors`: CHAZ predictors for historical events (from `bt_*.nc`). The predictors is calculated using monthly ERA-Interim reanalysis data. This file is used for calculating the seeding rate ratio for genesis model and for getting the regression stochastic error terms for intensity model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `landmask.nc`: landmask data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `result_l.nc`: regression parameters for near land cases

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `result_w.nc`: regression parameters for cases over open water




`pre`: the directory containing data used for preprocessing and data found in preprocessing

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `coefficient_meanstd.nc`: containing the mean and standard deviations of the predictors for historical events (i.e., those saved in bt_global_predictors.nc)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `Cov*YYYY.nc`: wind covariance of historical events

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `YYYY*r1i1p1f1.nc`:CMIP6 historic data used in `getMeanStd.py` to find mean and standard deviations of predictors for historical events

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `A_*YYYYMM.nc`: covariance, track, and time details of synthetic wind




`src`: contains all of the source code necessary to run CHAZ


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `calA.py`: calculate mean and covariance of synthetic wind


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `module_riskModel.py`: contains functions necessary to calculate genesis, track, and predictors


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `caldetsto_track_lysis.py`: return intensities based on calGenisi.py and module_GenBamPred.py output


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `preprocess.py`: preprocess historical and synthetic wind data


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `module_GenBamPred.py`: calculates tracks via Beta-Advection Model


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `calWindCov.py`: calculate historic wind mean and covariance in preprocessing




`tools`: contains all of the files with tools  necessary to run the source code of CHAZ.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `module_stochastic.py`: functions to find means and errors of predictors

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `readbst.py`: functions creating ibtracs and bst objects

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `regression4.py`: regression tools

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `util.py`: miscellaneous functions used in downscaling and preprocessing


## IV. CHAZ-pre

### 1) Download data, getting PI and TCGI

Preprocessing involves populating '\pre' wutg `r1i1p1_YYYY.nc`, `coefficientmeanstd.nc` and `A_YYYYMM.nc` from `calWindCov.py`, `calA.py`, `getMeanStd.py`, and `preprocess.py`. See examples for more details.

## V) Checking Initial conditions
Prior to conducting CHAZ downscaling, we should check the initial conditions those are nessary for CHAZ runs. They are TCGI_YYYY.mat, r1i1p1_YYYY.nc (from calPreprocess), A_YYYYMM.nc (from calA), coefficent_meanstd.nc (getMeanStd.py).

## VI) Running CHAZ (Downscaling)

### 1) Changing Global Variables in `Namelist.py`


`Model = 'ERAInterim'` - model that provides reanalysis data (str), default is European Center for Medium Range Weather Forecasts interim reanalysis

`ENS = 'r1i1p1'` - global model (str)

`TCGIinput = 'TCGI_CRH_SST'` - environmental parameters used as input for TCGI (str)

`CHAZ_ENS = 1` - number of track ensemble members per year (int)

`CHAZ__Int_ENS = 40` - number of intensity ensemble members (int)

`seedN = 1000` - annual seeding rate for random seeding (int). This function is currently under development.

`landmaskfile = 'landmask.nc'` - land mask file (NETCDF)

`ipath = ''` - location of environmental data TCGI, and mean/std of the predictors (str)

`opath = ''` - location of pik file with observed best track global predictors (str)

`obs_bt_path = ''` - location of observed best track data (str)

`Year1 = ` - the beginning year (int)

`Year2 = ` - tne end inner year (int)

`runCHAZ = ` - if `True`, CHAZ will run, if `False` CHAZ will not run (bool)

`calGen = ` - if `True`, genesis will  be calculated, if `False` genesis will not be calculated (bool)

`calBam = ` - if `True`, track will  be calculated, if `False` track will not be calculated (bool)

`calInt = ` - if `True`, intensity will  be calculated, if `False` intensity will not be calculated (bool)

### 2) CHAZ Downscaling

1. Make sure all the inputs are in the ipath, opath, and obs_bt_path 

2. Check that `output` is empty before running CHAZ

3. Simply type the following command:`$ ./CHAZ.py`


##  VII) Output of CHAZ

The output of CHAZ gives a set of netCDF files with names `[model name]_[year]_ens[ensemble number].nc` where "model name" is the model used (also specified in `Namelist.py`), "year" is the year of the simulation, and "ensemble number" is the ensemble number represented with three digits. 

The dimensions of each netCDF file are lifelength, stormID, and ensembleNum (the specific size of the dimensions can be found using the command `$ ncinfo [filename]`. "lifelength" is the amount length of time of the storm's life. "stormID" represents how many storms have been run at this year.  "ensembleNum" is the number of members in the intensity ensemble.

## Disclaimer

The output is not bias-corrected. Some bias correction may be necessary to estimate hurricane activity at the regional level.

## License

MIT License

Copyright (c) [2020] [The Columbia HAZard model (CHAZ) team]; The CHAZ team consists of Drs. Suzana J. Camargo, Chia-Ying Lee, Michael K. Tippett, and Adam H. Sobel from Columbia University. 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Publications


