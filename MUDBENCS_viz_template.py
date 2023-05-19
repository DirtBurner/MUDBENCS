# %% [markdown]
# # Data Analysis and Visualization Template - MUDBENCS
# This workbook demonstrates analysis and visualization tools developed to process data generated during the MUDBENCS project, and NSF funded initiative to sample the coastal regions of NE South America (French Guiana, Brazil) for water and sediment. 

# %%
#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MUDBENCS_calcs as mb


# %%
#Import Elemental Data from Aller Legacy Samples
df = mb.import_Aller_elements()
print('These are the columns in the dataframe that you may select to plot using the plot_element_ratios function:')
print(df.columns)


# %%
#Plot ratio cross plots of different elements

#Define which labels you would like to plot. Choose the X variable, Y variable, and normalizing variable (a quantity that forms the denominator of each axis):
x_lab = 'Al2O3'     #X variable
y_lab = 'K2O'       #Y variable
norm = 'SiO2'       #normalizing variable

#Make an element ratio plot of the data you specify for ratio cross plots
mb.plot_element_ratios(df, x_lab, y_lab, norm)

