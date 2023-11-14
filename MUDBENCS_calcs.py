#Functions for MUDBENCS Sample Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seabird.cnv import fCNV
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import glob

print('(((((((((((((((( MUDBENCS Date Analysis and Visualization Tools ))))))))))))))))')

DEBUG = False
OTS_stations = [
    'stn02',
    'stn04',
    'stn05',
    'stn06',
    'stn07',
    'stn08',
    'stn09',
    'stn10',
    'stn11',
    'stn12',
    'stn13',
    'stn14',
    'stn15',
    'stn16',
    'stn17',
    'stn18',
    'stn19',
    'stn20',
    'stn21'
]

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

    dat_df = cnv2df(dat_nc)

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

    #Creat axes for each variable in the profile.
    _, ax = plt.subplots(nrows=1, ncols=len(v))

    for z, axis in enumerate(ax.flatten()):
        profile_plotter(down_df, up_df, v[z], axis, z,  direction,)     
    

    return down_df, up_df, bounce_point, ax


def profile_plotter(down_df, up_df, v, ax, z, direction='both'):
    #Select downcast, upcast, or both dependent on keyword arguments.
    debug('Direction has been entered as ', direction)
    if direction == 'down':
        ax.plot(down_df[v], down_df['DEPTH'], 'b')
        debug('Plotting downcast...')
    if direction == 'up':
        ax.plot(up_df[v], up_df['DEPTH'], 'lightblue')
        debug('Plotting upcast...')
    if direction == 'both':
        ax.plot(down_df[v], down_df['DEPTH'], 'b')
        ax.plot(up_df[v], up_df['DEPTH'], 'lightblue')
        debug('Plotting both up and down casts...')
    if direction not in ['up', 'down', 'both']:
        print(direction, ' is not a valid choice. Choose up, down, or both. Plotting both casts.')
        ax.plot(down_df[v], down_df['DEPTH'], 'b')
        ax.plot(up_df[v], up_df['DEPTH'], 'lightblue')
        debug('Unrecognized kwarg for direction; plotting both up and down casts...')
    condition_axes(ax, v, z)
    debug('Function "profile_plotter has called "condition_axes" to label x-axis of subplot', z, 'as', v.title())

def condition_axes(ax, v, z):
    #Label axes for profile plotter, depending on the number (z) of the ax that is passed to it and the
    #variable name.
    if z == 0:
        debug('Function "condition_axes" is labeling the y-axis "Depth"')
        ax.set_ylabel('Depth (m)')
        ax.grid('True', axis='y')
    else:
        ax.grid('True', axis='y')
        ax.set_yticklabels([])
    ax.set_xlabel(v.title())
    ax.invert_yaxis()
    ax.ticklabel_format(axis='x', style='plain')

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

def MUDBENCS_map(data_df, variable, colormap='plasma', label_countries=True, add_stations=True, stationslist=OTS_stations, add_front=False):
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
            label_countries (boolean, default=True) - Add text labels to countries in the fixed map
            add_stations (Boolean, default=True) - Add points for coordinates of stations in stationslist
                The variable stationslist is hard wired in this .py file (scroll to top) to include all
                stations with over-the-side deployents and to ignore stations where observations (only)
                and/or through-water samples were taken. Any substitute list can be added, but the entries
                in the list have to match the "Station Number" values in MUDBENCS_Stations.csv or else 
                they simply will not be plotted.
            stationslist (list of strings, default=stationslist) - list of strings that correspond
                to station numbers (format: stnXX where XX is a two-digit station identifier with
                the exception of stn6_TurbEdge which was an observation site with no over-the-side
                deployments). If this list is manually input into the function call, and if any 
                strings are not in the column for station list, it will simply ignore those stations.
            add_front (Boolean, default=False) - add observed frontal boundary as an open square
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
    #Add optional map features called in kwargs:
    if label_countries == True:
        add_country_names(m)
    if add_stations == True:
        add_stations_list(m, stationslist)
    if add_front == True:
        add_front_site(m)
 

    return m

def add_country_names(m):
    '''
    This function is used conditionally (set by kwargs in calling function) in mapping functions to 
    add the country names manually into maps which show data from the cruise. The coordinates of
    the country names are based on the fixed coordinates within the function MUDBENCS_map as of 
    4 October 2023.


    '''

    lon = (-53, -53, -54.53)
    lat = (4.1, 0.5, 3.55)
    names = ['French Guiana', 'Brazil', 'Surin.']
    X,Y = m(lon,lat)
    for i, (X, Y) in enumerate(zip(X, Y), start=0):
        plt.text(X, Y, names[i], ha='center', color='k')

def add_stations_list(m, stationslist):
    '''
    Function called conditionally by MUDBENCS_map when a user enters keyword argument `add_stations=True`. 
    If the keyword argument stationslist=[] is used to enter a list of station names, only those stations
    will be plotted. If not, all stations in the hard-wired variable stationslist are included. These
    hardwired stations are the actual stations with over-the-side deployments, not frontal observation
    stations which were also numbered in station logs aboard the ship. Coordinates are stored in the 
    MUDBENCS_Stations.csv file in the repository.
    '''

    #Add all stations for which we have CTD data (all stations which do not contain the word "Test" in their names):
    stations = pd.read_csv('MUDBENCS_Stations.csv')
    #Plot only the stations that are in the stations list, passed from calling function or defaulting to var
    #defined in this .py file. 
    sample_stations = stations[stations['Station Number'].isin(stationslist)]
    debug(sample_stations['Station Number'])    
    #Add to plot:
    m.plot(sample_stations['Lon'], sample_stations['Lat'], latlon=True, marker='.', markersize=8, color='k', mec='white', linestyle='')

def add_front_site(m):
    '''
    Adds an observed front that was recorded in the station logs to the plot. The front was sharp enough
    to be picked up on radar in the bridge, so it is plotted as a square, open to avoid blocking the
    data underneath.
    '''

    front_lat = 2+20.3687/60
    front_lon = -(48+31.416/60)
    m.plot(front_lon, front_lat, latlon=True, marker='s', markersize=10, color='None', markeredgecolor='k')

def plot_one_to_one_line(ax):
    '''
    This simply adds a one to one line on a plot in black based on the axis limits of the ax handle passed 
    through. No outputs. 
    '''

    xes = ax.get_xlim()
    x = np.linspace(min(xes), max(xes), 10)
    y = x
    ax.plot(x, y, color='k')

def cnv2df(cnv_obj):
    '''
    Convert cnv files processed from Seabird hex files first into dictionary, then into a dataframe
    '''
    nc_dict = {}
    for key in cnv_obj.keys():
        debug(key, cnv_obj[key][0:5])
        debug(type(cnv_obj[key]))
        nc_dict |= {key:cnv_obj[key]}

    dat_df = pd.DataFrame(nc_dict)

    return dat_df

def bottle_depth_variables(bottle_file, down_df, up_df):
    bottle_depths = pd.read_csv(bottle_file, skiprows=2, header=None)
    
    debug('Shape of upcast dataframe: ', up_df.shape)
    debug('Upcast dataframe index range: ', up_df.index[0], ' to ', up_df.index[-1])

    bot_avgs = pd.DataFrame()
    bot_stds = pd.DataFrame()

    for ind, bot in bottle_depths.iterrows():
        bot_num = bot[0]
        debug('Bottle number: ', bot_num)
        depth_range = [float(up_df['DEPTH'][up_df.index == bot[3]].values), float(up_df['DEPTH'][up_df.index == bot[4]].values)]
        debug('Depth range: ', depth_range)
        bottle_depth_downcast_df = down_df.loc[(down_df['DEPTH'] <=  max(depth_range)) & (down_df['DEPTH'] >=  min(depth_range))]
        bottle_depth_averages, bottle_depth_stds = bottle_depth_downcast_df.mean(), bottle_depth_downcast_df.std()
        flag = 'Actual Values'
        #Check for NaNs, which are due to not enough distance between the beginning and end of a bottle firing. If
        #NaNs present, add distance in depth range:
        if bottle_depth_averages.isna().any():
            print('Nans present in bottle ', bot_num, '. Stretching depth range by 3%.')
            depth_range = [0.97*min(depth_range), 1.03*max(depth_range)]
            bottle_depth_downcast_df = down_df.loc[(down_df['DEPTH'] <=  max(depth_range)) & (down_df['DEPTH'] >=  min(depth_range))]
            bottle_depth_averages, bottle_depth_stds = bottle_depth_downcast_df.mean(), bottle_depth_downcast_df.std()
            flag = 'Stretched Depth Range (3%)'
        #Do it a second time if 3% is not enough, but add a fixed distance to the range
        if bottle_depth_averages.isna().any():
            print('Nans still present in bottle ', bot_num, '. Using upcast data instead.')
            bottle_depth_downcast_df = up_df.loc[(up_df['DEPTH'] <=  max(depth_range)) & (up_df['DEPTH'] >=  min(depth_range))]
            bottle_depth_averages, bottle_depth_stds = bottle_depth_downcast_df.mean(), bottle_depth_downcast_df.std()
            flag = 'Used upcast data due to depth differences from downcast data.'
        
        #Add bottle numbers to each series and compile a dataframe with the series:
        bottle_depth_averages['Bottle Number'] = bot_num
        bottle_depth_stds['Bottle Number'] = bot_num
        bottle_depth_averages['Depth avg flag'] = flag
        bottle_depth_stds['Depth avg flag'] = flag
        bot_avgs = bot_avgs.append(bottle_depth_averages, ignore_index=True)
        bot_stds = bot_stds.append(bottle_depth_stds, ignore_index=True)
        
        debug('Loop ', ind, bottle_depth_averages, bottle_depth_stds)
        debug('\n', '\n')

    return bot_avgs, bot_stds


def load_CTD_data(CTD_num, variables = ['density', 'TEMP', 'PSAL'], plot_data=True, direction='both'):
    file_name, bottle_file = find_files(CTD_num)
    dat_nc = get_CNV_data(file_name)
    
    #Print attributes such as latitude and longitude, as well as time measured. Uses the output file from above. 
    print('The profile coordinates are latitude: %.4f, and longitude: %.4f, collected at ' % (dat_nc.attributes['LATITUDE'], dat_nc.attributes['LONGITUDE']), dat_nc.attributes['gps_datetime'])
    print('Data types available are: ', dat_nc.keys())

    #Get up and down cast data and plot the chose variables
    down_df, up_df = separate_up_down(dat_nc, find_bottom(dat_nc))
    
    #Get bottle averages
    bot_avgs, bot_stds = bottle_depth_variables(bottle_file, down_df, up_df)

    #Plot if desired
    if plot_data == True:
        _, _, _, ax = plot_profile(dat_nc, variables, direction='down')
    elif plot_data == False:
        ax = None

    return down_df, up_df, bot_avgs, bot_stds, ax


def find_files(CTD_num):
    for file in glob.glob('*'+CTD_num+'*.cnv'):
        file_name = file
    for bl_file in glob.glob('*'+CTD_num+'*.bl'):
        bot_file_name = bl_file

    return file_name, bot_file_name
