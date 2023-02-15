import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LoadData import file_names

df = pd.read_pickle('dataframe.pkl')

# Weighted average precipitation

areasP = np.array([865216821,665778630,383284467.2,785339562.5, 131465617.03602804, 170280882.42145243, 150530436.137325912714005])
Pcatchment = (areasP[0]*df['rain_CHHA']+areasP[1]*df['rain_DCCT']+areasP[2]*df['rain_MMMO']+areasP[3]*df['rain_NMKI']
              +areasP[4]*df['rain_SPPT']+areasP[5]*df['rain_TGLG']+areasP[6]*df['rain_WCHN'])/np.sum(areasP)
Paverage = np.average(Pcatchment)*365 #mm/year

# Weighted average evaporation

areasPET = np.array([632073506, 443472572, 2017865143, 697936853 ])
PETcatchment = (areasPET[0]*df['PET_328201']+areasPET[1]*df['PET_328202']+areasPET[2]*df['PET_330201']
                +areasPET[3]*df['PET_351201'])/np.sum(areasPET)
PETaverage = np.average(PETcatchment)*365 #mm/year