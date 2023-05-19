#Functions for MUDBENCS Sample Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('(((((((((((((((( MUDBENCS Date Analysis and Visualization Tools ))))))))))))))))')

DEBUG = False

def import_Aller_elements():
    '''
    Import inorganic elemental data from CNRS measured in spring 2023 on Aller legacy samples. 
        Inputs:
            none - hard wired to open the Aller data measured in Spring 2023. 
    '''
    filename = 'Rosenheim 9758 230511.xls'

    #Use pandas excel read and skip rows with text only. Row 12 is skipped because it only contains the units. All units are
    #% except Sc measurements, which are microgram per gram. 
    df = pd.read_excel(filename, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],)

    return df

def plot_element_ratios(df, x_lab, y_lab, norm):
    '''
    This function plots A/B on C/B, where A is the y-axis element, C is the x-axis element, and B normalizes both elements. The function
    produces a scatter plot with a linear fit through the data. 
        Inputs:
            df (dataframe): This is the dataframe output of import_Aller_elements
            x_lab (string): This is the x_label of the graph you will produce, and must match one of the colum names in df
                default: 'Al2O3'
            y_lab (string): This is the y_label of the graph you will produce, and must match one of the colum names in df
                default: 'K2O'
            norm (string): This is the normalizing element that will be the denominator of both axes. It must match the 
                column names of df. Default: 'SiO2'

    '''
    #Check for instances where bad inputs necessitate default values:
    if x_lab not in df.columns:
        x_lab = 'Al2O3'
        print('!!! x_lab not recognized, default to Al2O3. Please check the column names and try again with exact match')
    if y_lab not in df.columns:
        y_lab = 'K2O'
        print('!!! y_lab not recognized, default to K2O. Please check the column names and try again with exact match')
    if norm not in df.columns:
        norm = 'SiO2'
        print('!!! norm not recognized, default to SiO2. Please check the column names and try again with exact match')
    

    df = df.sort_values(by=x_lab) 
    x = [a/b for a,b in zip(df[x_lab], df[norm])]
    y = [c/d for c,d in zip(df[y_lab], df[norm])]
    _, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.plot(
        x,
        y,
        mec='k',
        mfc='peru',
        marker='o',
        markersize=8,
        linestyle=''
    )
    ax1.set(xlabel=x_lab+'/'+norm, ylabel=y_lab+'/'+norm)
    m, b = np.polyfit(x, y, 1)
    model = [val*m+b for val in x]
    ax1.plot(
        x,
        model,
        linestyle='--',
        color='k'
    )
    

    ax1.text(1.1*ax1.get_xlim()[0], 0.9*ax1.get_ylim()[1], '('+y_lab+'/'+norm+') = '+'('+x_lab+'/'+norm+')x'+str(round(m, 3))+' + '+str(round(b, 3)))

