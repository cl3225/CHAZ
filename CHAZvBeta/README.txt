The Columbia HAZard model(CHAZ)
------------------------------------------------------
Chia-Ying Lee (cl322@columbia.edu)
Emma Schechter (es3522@columbia.edu) 7/17/21

-------------------------------------------------------
Overview of CHAZvBeta contents; preprocessing is flagged as 'True', 'output' has yet to be populated, and 'pre' has yet to be populated with r1i1p1_YYYY.nc, coefficientmeanstd.nc and A_YYYYMM.nc from preprocessing. 

- CHAZ.py: Main script for running CHAZ model and prepare the input data

- Namelist.py: Script of global variables and control  

- getMeanStd.py: Script for calculating the mean and standard deviation of the intensity model predictors. This subroutine will be mearge to src in the future. 

- inputs: Directory containing default CHAZ input from model development. Note: you will only need to change contents if you are re-fitting the CHAZ model

- pre: Directory containing the data generated from preprocessing

- outputs: Directory containing CHAZ output, i.e., synthtic storm events 

- src: Directory containing source code 


Detail instructions please see: https://cl3225.github.io/CHAZ.github.io/ 

- tools: Directory containing tools used in source code, CHAZ.py, and Namelist.py



