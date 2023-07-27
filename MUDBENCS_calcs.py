#Functions for MUDBENCS Sample Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seabird.cnv import fCNV
from mpl_toolkits.basemap import Basemap
import geopandas as gpd

print('(((((((((((((((( MUDBENCS Date Analysis and Visualization Tools ))))))))))))))))')

DEBUG = False

def debug(*args):
    if DEBUG == True:
        print(args)

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

def get_CNV_data(cnv_file):
    '''
    Get the data from CNV file into a masked python array.
    '''

    dat_nc = fCNV(cnv_file)

    return dat_nc

def find_bottom(dat_nc):
    '''
    Find the index of the bottom of the CNV file. 
    '''
    
    bounce_point = np.argmax(dat_nc['DEPTH'])

    return bounce_point

def separate_up_down(dat_nc, bounce_point):
    '''
    Function to separate the up and down cast of a typical CNV file from Seabird. Function makes a 
    dictionary out of the keyed arrray, turns that to a DataFrame, and uses Panda slicing to 
    create two dataframes.
    '''

    cols = dat_nc.keys()
    dat_dict = {}
    for column in cols:
        dat_dict |= {column:dat_nc[column]}

    dat_df = pd.DataFrame(dat_dict)

    dat_df_down, dat_df_up = dat_df.loc[dat_df.index<bounce_point], dat_df.loc[dat_df.index>bounce_point]

    return dat_df_down, dat_df_up

def plot_profile(dat_nc, variable, direction='down'):
    '''
    Function takes a data array returned by the seabird.cnv module and produces a plot of a single
    variable from the specified CTD cast. The user can choose down cast, upcast, or both. Default
    is downcast data.


    '''

    bounce_point = find_bottom(dat_nc)
    down_df, up_df = separate_up_down(dat_nc, bounce_point)

    if type(variable) != list:
        variable = [variable]

    
    v = []

    for item in variable:
        temp = variable_checker(item, dat_nc)
        v.append(temp)
        debug('Building out the variables from variable. Adding:', v)

    cols = len(v)

    _, ax = plt.subplots(nrows=1, ncols=cols)

    for z, axis in enumerate(ax.flatten()):
        profile_plotter(down_df, up_df, v[z], direction, axis)
        debug('v[z].title is ', v[z].title())
        plt.xlabel(v[z].title())
        axis.invert_yaxis()
        if z == 0:
            debug('Labeling y-axis with', z, v[z])
            plt.ylabel('Depth')
    

    return down_df, up_df, bounce_point, ax


def profile_plotter(down_df, up_df, v, direction, ax):
    #Select downcast, upcast, or both dependent on keyword arguments.
    if direction == 'down':
        ax.plot(down_df[v], down_df['DEPTH'], 'b')
    if direction == 'up':
        ax.plot(up_df[v], up_df['DEPTH'], 'lightblue')
    if direction == 'both':
        ax.plot(down_df[v], down_df['DEPTH'], 'b')
        ax.plot(up_df[v], up_df['DEPTH'], 'lightblue')
    if direction not in ['up', 'down', 'both']:
        print(direction, ' is not a valid choice. Choose up, down, or both. Plotting both casts.')
        ax.plot(down_df[v], down_df['DEPTH'], 'b')
        ax.plot(up_df[v], up_df['DEPTH'], 'lightblue')

def variable_checker(variable, dat_nc):
    # Check keyword arguments
    if 'DEPTH' in dat_nc.keys():
        if variable in dat_nc.keys():
            v = variable
        else:
            
            print('-?-?-?-?-?-?-?-?')
            print(variable, ' is not recorded in this data file. Either re-process your data in SeaSoft or select a new variable!')
            print('Available variables are: ', dat_nc.keys())
            if 'TEMP' in dat_nc.keys():
                v = 'TEMP'
                print('Plotting depth profile of temperature instead of ', variable)
            else:
                v = dat_nc.keys()[0]
                print('Plotting depth profile of ', v)
    else:
        print('Depth is not recorded in this CNV file. Please reprocess data in SeaSoft and include the variable depth!')
        return
    
    return v

def read_along_track_data(begin_end):
    '''
    Along track data from multiple systems is recorded onboard the ship while it is underway. This
    takes the data in as a dataframe. Set DEBUG to true to print the column names.

        Inputs:
            begin_end (string or list of strings) - Can be Leg 1, Leg 2, Both, or a list of two strings
                depicting in the format DD-MMM-YYYY HH:mm where time is on the 24 hour clock 
                (not the 12 hour clock). The first should be the beginning time stamp and the second 
                should be the end time stamp
        Outputs:
            AT_data_df (DataFrame) - pandas dataframe with the data
    '''

    #Read the data from .dat file and parse the time and date together
    AT_data_df = pd.read_csv(
        'WS23139_Rosenheim-Full Vdl.dat', #File is hardwired here, included in repository
        skiprows=1,
        delimiter='\t',
        parse_dates=[['Date', 'Time']]
    )

    #Mask for beginning and end of data (data subset) and convert to pandas datetime format:
    #Select defaults
    mask_dates = []
    if type(begin_end) == str:
        if begin_end == 'Leg 1':
            mask_dates = ['6-Jun-2023 16:45', '12-Jun-2023 16:00']
        if begin_end == 'Leg 2':
            mask_dates = ['14-Jun-2023 14:15', '17-Jun-2023 16:00']
        if begin_end == 'Both':
            mask_dates = ['6-Jun-2023 16:45', '17-Jun-2023 16:00']
        else:
            print('You entered begin_end as ', begin_end, '. Expected Leg 1, Leg 2, or Both, so masking for Leg 1 by default.')

    else:
        mask_dates = begin_end
        
    AT_data_df = mask_expedition_legs(AT_data_df, mask_dates)
    pd.to_datetime(AT_data_df['Date_Time'])
    debug(AT_data_df.columns)

    #Create a list of coordinate columns that need to converted from strings to floats for plotting:
    col_list = [
        'Lon Dec. Deg.', 
        'Lat Dec. Deg.',
        'Lon Dec. Deg..1', 
        'Lat Dec. Deg..1',
        'Lon Dec. Deg..2',
        'Lat Dec. Deg..2'       
    ]
    convert_str_coords_by_broadcast(AT_data_df, col_list)

    return AT_data_df

def convert_str_coords_by_broadcast(data_df, column):
    '''
    This function takes the specified columns and converts them to floating point numbers from
    strings. It is designed particularly for the along track data of the Walton Smith, which supplies
    GPS coordinates as strings with leading spaces and spaces between the sign (+ or -) and the numeric value,
    which is also a string. This function may be too slow! 

        Inputs:
            data_df (DataFrame): this is a dataframe, and ideally should be the dataframe output within
                the 'read_along_track_data' function.
            column (string, or list of strings): string that specifies column to be converted. Function should
                test whether this column is a string
    '''
    #Check if column is a list of strings or a single string. If single string, make it a one-member list
    if type(column) != list:
        debug('not a list!')
        column = [column]

    for col in column:
        data_df[col]=data_df[col].str.replace(' ','').astype(float)

    return data_df

def mask_expedition_legs(data_df, begin_end):
    '''
    Function to mask the along track data for stations or legs.
        Inputs:
            data_df (DataFrame) - data which has some columns with annoying coordinate strings containing
                internal spaces and leading spaces. This is the output of reading the data into Python
            begin_end (list of strings) - This should be a list of date/time strings in the format 
                DD-MMM-YYYY HH:mm where time is on the 24 hour clock (not the 12 hour clock). The first
                should be the beginning time stamp and the second should be the end time stamp
    '''
    if len(begin_end) > 2:
        print('More than two timestamps to define data subset! Using only the first two')
    if len(begin_end) < 2:
        begin_end = ['6-Jun-2023 16:45', '12-Jun-2023 16:00']   #Leg 1 timestamp is the default
        print('Less than two time points entered! Using defaults instead for Leg 1\n 6-Jun-2023 16:45 to 12-Jun-2023 16:00')
    begin, end = pd.to_datetime(begin_end[0]), pd.to_datetime(begin_end[1])
    leg_data_df = data_df.loc[(data_df['Date_Time']>=begin) & (data_df['Date_Time']<=end)]

    return leg_data_df

def MUDBENCS_map(data_df, variable, colormap='plasma'):
    '''
    This function creates a Lambert Conformal projection with continents, shaded relief, and 
    outlines of countries. The function plots the points recorded by the ship underway data system 
    colored with the variable of choice to display changes in that variable over space. 
    
        Inputs:
            data_df (DataFrame) - this is the output of read_along_track data, masked for different
                legs of the cruise of times on/between stations.
            variable (string) - the exact name of the column with data you wish to be the source of 
                color of the points along track. If you have trouble finding the exact name of a 
                column, run read_along_track data with MB.DEBUG=True, or give the command 
                print(data_df.columns). 
            Keyword Arguments In
            color_map (string, default color_map='plasma') - color map for the plotted points
        Outputs:
            m (map axes) - the map axes, for adding more annotation or points to the map

    '''
    #Check that variable requested is in the columns of the data_df
    if variable in data_df.columns:
        variable = variable
    else:
        variable = ' Salinity PSU'
        print('Variable not recognized. Plotting salinity by default. Try again by listing all column names.')

    
    fig = plt.figure()
    m = Basemap(projection='lcc', resolution='h', lat_0=2.5, lon_0=-51, width=0.9e6, height=0.9e6)
    m.shadedrelief()
    m.drawcoastlines(color='white', linewidth=1)
    m.drawcountries(color='lightgray', linewidth=1)
    m.drawparallels(np.arange(-80., 81., 5.), labels=[True,False,False,True])
    m.drawmeridians(np.arange(-180., 181., 5.), labels=[True,False,False,True])
    x, y = m(data_df['Lon Dec. Deg.'], data_df['Lat Dec. Deg.'])
    m.scatter(x, y, c=data_df[variable], s=25, cmap=colormap)
    plt.colorbar( label=variable.strip())
   

    return m
