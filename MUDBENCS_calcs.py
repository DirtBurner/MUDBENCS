#Functions for MUDBENCS Sample Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seabird.cnv import fCNV
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import glob
import cycler as cy
import matplotlib as mpl

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

    #Create axes for each variable in the profile.
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

    
    map_ax = Basemap(projection='lcc', resolution='h', lat_0=2.5, lon_0=-51, width=0.9e6, height=0.9e6)
    map_ax.shadedrelief()
    map_ax.drawcoastlines(color='white', linewidth=1)
    map_ax.drawcountries(color='lightgray', linewidth=1)
    map_ax.drawparallels(np.arange(-80., 81., 5.), labels=[True,False,False,True])
    map_ax.drawmeridians(np.arange(-180., 181., 5.), labels=[True,False,False,True])
    x, y = map_ax(data_df['Lon Dec. Deg.'], data_df['Lat Dec. Deg.'])
    m = map_ax.scatter(x, y, c=data_df[variable], s=25, cmap=colormap)
    #if ax:
    #    plt.colorbar(label=variable.strip(), ax=ax, mappable=m)
    #else:
    #    plt.colorbar(label=variable.strip())
    plt.colorbar(label=variable.strip())
    #Add optional map features called in kwargs:
    if label_countries == True:
        add_country_names(map_ax)
    if add_stations == True:
        add_stations_list(map_ax, stationslist)
    if add_front == True:
        add_front_site(map_ax)
 

    return map_ax

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
        debug('Bottle Depth averages and standard deviations: ', bottle_depth_averages, bottle_depth_stds)
        flag = 'Actual Downcast Values'
        #Check for NaNs, which are due to not enough distance between the beginning and end of a bottle firing. If
        #NaNs present, add distance in depth range:
        if bottle_depth_averages.isna().any():
            print('Nans present in bottle ', bot_num, '. Stretching depth range by 3%.')
            depth_range = [0.97*min(depth_range), 1.03*max(depth_range)]
            bottle_depth_downcast_df = down_df.loc[(down_df['DEPTH'] <=  max(depth_range)) & (down_df['DEPTH'] >=  min(depth_range))]
            bottle_depth_averages, bottle_depth_stds = bottle_depth_downcast_df.mean(), bottle_depth_downcast_df.std()
            debug('Bottle Depth averages and standard deviations: ', bottle_depth_averages, bottle_depth_stds)
            flag = 'Stretched Depth Range (3%), Downcast'
        #Do it a second time if 3% is not enough, but add a fixed distance to the range
        if bottle_depth_averages.isna().any():
            print('Nans still present in bottle ', bot_num, '. Using upcast data instead.')
            bottle_depth_downcast_df = down_df.loc[(down_df['DEPTH'] <=  max(depth_range)) & (down_df['DEPTH'] >=  min(depth_range))]
            bottle_depth_averages, bottle_depth_stds = bottle_depth_downcast_df.mean(), bottle_depth_downcast_df.std()
            debug('Bottle Depth averages and standard deviations: ', bottle_depth_averages, bottle_depth_stds)
            flag = 'Used Upcast Data Due to Depth Differences from Downcast Data.'
        
        #Add bottle numbers to each series and compile a dataframe with the series:
        bottle_depth_averages['Bottle Number'] = bot_num
        bottle_depth_stds['Bottle Number'] = bot_num
        bottle_depth_averages['Depth avg flag'] = flag
        bottle_depth_stds['Depth avg flag'] = flag
        bot_avgs = pd.concat([bot_avgs, bottle_depth_averages], ignore_index=True)
        bot_stds = pd.concat([bot_stds, bottle_depth_stds], ignore_index=True)
        
        debug('Loop ', ind, bottle_depth_averages, bottle_depth_stds)
        debug('\n', '\n')

    return bot_avgs, bot_stds


def load_CTD_data(CTD_num, variables = ['density', 'TEMP', 'PSAL'], plot_data=True, direction='both', up_cast_only = False):
    file_name, bottle_file = find_files(CTD_num)
    dat_nc = get_CNV_data(file_name)
    
    #Print attributes such as latitude and longitude, as well as time measured. Uses the output file from above. 
    print('The profile coordinates are latitude: %.4f, and longitude: %.4f, collected at ' % (dat_nc.attributes['LATITUDE'], dat_nc.attributes['LONGITUDE']), dat_nc.attributes['gps_datetime'])
    print('Data types available are: ', dat_nc.keys())

    #Get up and down cast data and plot the chose variables
    down_df, up_df = separate_up_down(dat_nc, find_bottom(dat_nc))
    
    #Get bottle averages
    if up_cast_only == True:
        bot_avgs, bot_stds = bottle_depth_variables_up_only(bottle_file, up_df)
        print('Upcast data selected!')
    else:
        bot_avgs, bot_stds = bottle_depth_variables(bottle_file, down_df, up_df)

    

    #Plot if desired
    if plot_data == True:
        _, _, _, ax = plot_profile(dat_nc, variables, direction=direction)
    elif plot_data == False:
        ax = None

    return down_df, up_df, bot_avgs, bot_stds, ax


def find_files(CTD_num):
    for file in glob.glob('*'+CTD_num+'*.cnv'):
        file_name = file
    for bl_file in glob.glob('*'+CTD_num+'*.bl'):
        bot_file_name = bl_file

    return file_name, bot_file_name


def bottle_depth_variables_up_only(bottle_file, up_df):
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
        bottle_depth_downcast_df = up_df.loc[(up_df.index <=  bot[4]) & (up_df.index >=  bot[3])]
        bottle_depth_averages, bottle_depth_stds = bottle_depth_downcast_df.mean(), bottle_depth_downcast_df.std()
        flag = 'Actual Upcast Values'
             
        
        #Add bottle numbers to each series and compile a dataframe with the series:
        bottle_depth_averages['Bottle Number'] = bot_num
        bottle_depth_stds['Bottle Number'] = bot_num
        bottle_depth_averages['Depth avg flag'] = flag
        bottle_depth_stds['Depth avg flag'] = flag
        bot_avgs = pd.concat([bot_avgs, bottle_depth_averages], ignore_index=True)
        bot_stds = pd.concat([bot_stds, bottle_depth_stds], ignore_index=True)
        
        debug('Loop ', ind, bottle_depth_averages, bottle_depth_stds)
        debug('\n', '\n')

    return bot_avgs, bot_stds

def MC_core_profiles(df, core='17MC', variables=['C%', 'N%', 'd13C', 'C:N (mass)', 'd15N', '%CaCO3'], inflator=0.2, col='peru'):
    '''
    Plots vertical profiles of multicore general geochem data. 

        Inputs:
            df (dataframe): single dataframe containing general geochemistry data as columns and individual
                core depths as rows. Cores are identified in each row, so the dataframe can be sliced by core.
            core (string): string specifying the core number, in format used during MUDBENCS cruise. Example:
                '04MC' is the 4th sequential multicore taken. That is the only string needed. Function works
                on one core at a time. 
            variables (list of string): This list should consist of strings that correspond to the column 
                headers of the dataframe input. If in doubt, please use the command `print(df.columns)` to 
                get a list of columns that can be entered into this list. A "magic" column exists through
                logic in this call; if a user enters `%CaCO3` into the list, the function will subtract the
                %C of the carbon run (acid treated) from the %C of the nitrogen run (not acid treated) to 
                provide a downcore plot of an estimate of the amount of carbonate mineral that was removed.
            inflator (float, default 0.2): Number to control the axes labels indirectly. This number is a proportion of the 
                x-axis range that is added to the full range so that the axes spread away from the min and max 
                a bit. One can also use the axs output to do this manually for every subplot, but I have found 
                this works well. Change to 0 to suppress this control.
            col (string, default 'peru'): Color of points in the plot. Please take care to use a pyplot color
                string, others will generate a line property error.

        Outputs:
            ax (pyplot axis object): axis object to do small modifications to the plots.
    '''
    if type(variables) is list:
        variables = variables
    else:
        variables = [variables] #Protects against single variable entry rendering the loop inoperable.

    #Create axes in the figure
    _, axs = plt.subplots(1, len(variables))
    df = df.sort_values(by='Top Depth')

    for j, variable in enumerate(variables):
        center_depth = (df[df['Core']==core]['Top Depth']+df[df['Core']==core]['Bottom Depth'])/2
        axs[j].plot(
            df[df['Core']==core][variable],
            -center_depth,
            marker='o',
            color=col,
            mec='k',
            linestyle='--'
        ) 
        #Define universal x-limits so that all plots are on the same axes
        infl = inflator  #inflates x-axis by a proportion. 0 means no inflation.
        xlimits = [min(df[variable])-infl*abs(min(df[variable])), max(df[variable])+infl*abs(max(df[variable]))]
        debug(variable, 'x-axis limits = ', xlimits)
        axs[j].set_xlim(xlimits)
        if variable == 'd15N':
            varname = r'$\delta^{15}$N'
        elif variable == 'd13C':
            varname = r'$\delta^{13}$C'
        elif variable == '%CaCO3':
            varname = r'% CaCO$_3$'
        else:
            varname = variable
        axs[j].set(xlabel=varname)
        if j == 0:
            axs[j].set(ylabel='Depth (cm)')
        else:
            axs[j].set(yticks=[])

    return axs

def load_organic_geochem_MC():
    '''
    Loads multicore organic geochemistry data from a fixed spreadsheet that contains all results from all cores
    including repeated analyses.

    '''

    carbon_df = pd.read_excel('Mudbencs MC Carbon compiled.xlsx', 'Sheet1', skiprows=20)
    debug(carbon_df.columns)
    C_df = carbon_df[['Identifier 1', 'd15N (‰, AT-Air)', 'd13C (‰, VPDB)', 'N%', 'C%', 'C:N (mass)', 'N (µmol)', 'C (µmol)']]
    debug(C_df.shape)
    nitrogen_df = pd.read_excel('Mudbencs MC Nitrogen compiled.xlsx', 'Sheet1', skiprows=20)
    N_df = nitrogen_df[['Identifier 1', 'd15N (‰, AT-Air)', 'N%', 'C%', 'N (µmol)']]
    debug(N_df.shape)
    CN_df = N_df.merge(C_df, on='Identifier 1', how='outer', indicator=True, suffixes=('(N)', None))
    debug(CN_df.shape)

    #Rename the columns for ease of use. 
    CN_df.rename(columns={
        "Identifier 1": "Sample ID",
        "d15N (‰, AT-Air)(N)": "d15N",
        "N (µmol)(N)":"umols N (N)",
        "d15N (‰, AT-Air)":"d15N (C)",
        "d13C (‰, VPDB)":"d13C",
        "N (µmol)":"umols N",
        "C (µmol)":"umols C"
    }, inplace=True)

    #Separate the Sample ID column into useful core numbers, top depths, and bottom depths.
    top_depth_list = []
    bottom_depth_list = []
    core_list = []
    for n, row in CN_df.iterrows():
        core = row['Sample ID'].split('-')[2]
        top = float(row['Sample ID'].split(' ')[1].split('c')[0].split('-')[0])
        bottom = float(row['Sample ID'].split(' ')[1].split('c')[0].split('-')[1])
        top_depth_list.append(top)
        bottom_depth_list.append(bottom)
        core_list.append(core)

    CN_df['Top Depth'] = pd.Series(top_depth_list)
    CN_df['Bottom Depth'] = pd.Series(bottom_depth_list)
    CN_df['Core'] = pd.Series(core_list)
    #Add %CaCO3 column from calculation involving the C% columns from both of the merged dataframes:
    CN_df['%CaCO3'] = CN_df['C%(N)'] - CN_df['C%']
    #Correct the C% column to reflect the missing mass of carbonate minerals from the carbon runs:
    CN_df['C%'] = CN_df['C%']*(1-CN_df['%CaCO3'])

    debug(CN_df.head())

    print(
        'Dataframe formed from merger of two spreadsheets. Pandas merge function creates a column "_merge"\n',
        'that tells you left, right, or both. This is an important column to check. The "left" dataframe\n',
        'contains carbon run analyses, and the right one contains nitrogen run analyses. Verify that \n',
        'any rows with either left or right in the _merge column are unique to either the C or N dataframe.'
    )

    return CN_df

def plot_MCs(og_df, corelist, variable, cmap='plasma', shape_list=['o', '^', 's', 'd']):
    '''
    Plots the vertical profiles of a variable in several cores which are supplied by a core list

        Inputs:
            og_df (dataframe): The dataframe produced by the function load_organic_geochem_MC
            corelist (list of strings): A list of multicores in format XXMC were XX is the two-digit 
                sequential core number. The routine works with slices of og_df, so entering a non-
                sensical core number will result in an entry in the legend with no data in the plot
                but it will not throw an error. 
            variable (string): This is a string that is identical to one of the columns in og_df. The 
                variable must be an exact match, so print out the columns of og_df (`print(og_df.columns)`)
                prior to using this function, or if you get key errors.
            cmap (string): Corresponds to colormap from matplotlib. Accessed through matplotlib.colors.get_cmap,
                thus is must be a matplotlib recognized colormap. Default is plasma.
            shape_list (list of strings): Provides a list of shapes to cycler. Cycler recognizes matplotlib 
                shapes, so shapes must be of that ilk. This cycles through the shapes with different colors
                so that individual cores will have a unique shape and color instance. Default is circle,
                triangle (upward), square, diamond. You may want to add more, have less, or simply change
                shapes for graph readability. 
        
        Outputs:
            ax (matplotlib axis handle): the axis handle so that you can make small modifications to the axes
    '''

    #Set shape and color list for cycling through the corelist:
    norm = mpl.colors.Normalize(vmin=0, vmax=len(corelist))
    cmap = mpl.cm.get_cmap(cmap)
    color_list = [cmap(norm(ind)) for (ind, core) in enumerate(corelist)]


    fig, ax = plt.subplots(nrows=1, ncols=1)
    custom_cycler_shape = (cy.cycler(marker=shape_list))
    ax.set_prop_cycle(custom_cycler_shape)

    #Set x-variable:


    #Set variable label:
    if variable == 'd15N':
        varname = r'$\delta^{15}$N'
    elif variable == 'd13C':
        varname = r'$\delta^{13}$C'
    elif variable == '%CaCO3':
        varname = r'% CaCO$_3$'
    else:
        varname = variable

    for j, core in enumerate(corelist):
        ax.plot(og_df[og_df['Core']==core][variable], -og_df[og_df['Core']==core]['Top Depth'], linestyle='', mec='k', color=color_list[j], markersize=10)

    ax.set(xlabel=varname, ylabel='Depth (cm)')
    ax.legend(corelist)

    return ax


def nuts_coords(nuts_df ):
    '''
    Opens the coordinates file, then searches the nuts_df for station numbers. Matches station numbers and 
    adds coordinates to the dataframe with columns names that allow compatability with MUDBANCS_map function
    
    '''

    stations = pd.read_csv('MUDBENCS_Stations.csv')
    debug(stations.columns)

    stations_split = [station.split(' ')[1] if ('MUDBENCS' in station) else station.split('-')[1][0:2] for station in nuts_df['Station']]
    station_type = ['Fish Surface' if ('MUDBENCS' in station) else 'Other' for station in nuts_df['Station']]
    stations_corr = ['0'+station if len(station)<2 else station for station in stations_split]
    debug(stations_corr)

    nuts_df['Station Number'] = stations_corr
    nuts_df['Station Type'] = station_type
    nuts_df['Lat Dec. Deg.'] = np.nan
    nuts_df['Lon Dec. Deg.'] = np.nan
    for n, row in nuts_df.iterrows():
        mask = [True if (row['Station Number'] in number) else False for number in stations['Station Number']]
        #print(stations.loc[mask])
        nuts_df['Lat Dec. Deg.'][n] = stations['Lat'].loc[mask]
        nuts_df['Lon Dec. Deg.'][n] = stations['Lon'].loc[mask]

    return nuts_df

def MC_tops_coords(MC_tops_df):
    '''
    Opens the coordinates file, then searches the nuts_df for station numbers. Matches station numbers and 
    adds coordinates to the dataframe with columns names that allow compatability with MUDBANCS_map function
        
    '''

    stations = pd.read_csv('MUDBENCS_Stations.csv')
    debug(stations.columns)

    stations_split = [station.split(' ')[0] for station in MC_tops_df['Station']]
    stations_split = [station.split('N')[1] for station in stations_split]
    stations_corr = ['0'+station if len(station)<2 else station for station in stations_split]
    print(stations_corr)

    MC_tops_df['Station Number'] = stations_corr

    MC_tops_df['Lat Dec. Deg.'] = np.nan
    MC_tops_df['Lon Dec. Deg.'] = np.nan
    for n, row in MC_tops_df.iterrows():
        mask = [True if (row['Station Number'] in number) else False for number in stations['Station Number']]
        #print(stations.loc[mask])
        MC_tops_df['Lat Dec. Deg.'][n] = stations['Lat'].loc[mask]
        MC_tops_df['Lon Dec. Deg.'][n] = stations['Lon'].loc[mask]

    return MC_tops_df

def load_nuts():
    '''
    Function to load nutrients data, parsing the data from the Buck Lab (University of South Florida) into 
    several DataFrames for use in plotting. This routine also parses the station labeling routine from 
    Calyn Crawford and Brenna Boehman which deviated from the perscripted labeling plan in order to apply
    coordinates to the data. Coordinates are added in another function.

    *The file that is hardwired here was updated with nutrient data from 2 reruns of the data that was sent from
    Kristen Buck to Brad Rosenheim on April 17, 2024. Original file contained cells that had '<LOD' if the sample
    was below detection limit; these data just contained 0.00 as a value. 
    
    '''


    debug('Accessing nutrients file MUDBENCS_2023_Nuts_Results V2.xlsx...')
    nuts_df = pd.read_excel('MUDBENCS_2023_Nuts_Results v2.xlsx', skiprows=4, usecols='A:E', nrows=46, names=['Station', 'P', 'N+N', 'Si Acid', 'Nitrite'], header=None)
    nuts_df = nuts_df.replace([0.00], [np.nan]) #Not sure why this has to be a list, but it doesn't work if not set up as a list.   

    QC_df = pd.read_excel('MUDBENCS_2023_Nuts_Results v2.xlsx', skiprows=4, usecols='H:L', nrows=11, names=['Standard', 'P', 'N+N', 'Si Acid', 'Nitrite'])
    reference_df = pd.read_excel('MUDBENCS_2023_Nuts_Results v2.xlsx', skiprows=4, usecols='O:S', nrows=7, names=['Standard', 'P', 'N+N', 'Si Acid', 'Nitrite'])
    RMNS_df = pd.read_excel('MUDBENCS_2023_Nuts_Results v2.xlsx', skiprows=4, usecols='V:Z', nrows=21, names=['Standard', 'P', 'N+N', 'Si Acid', 'Nitrite'])
    LOD_df = pd.read_excel('MUDBENCS_2023_Nuts_Results v2.xlsx', skiprows=17, usecols='P:S', nrows=1, names=['P', 'N+N', 'Si Acid', 'Nitrite'])
    MC_coretop_df = pd.read_excel('MUDBENCS_2023_Nuts_Results v2.xlsx', skiprows=53, usecols='A:E', nrows=12, names=['Station', 'P', 'N+N', 'Si Acid', 'Nitrite']).dropna(axis=1)
    
    print('Values for all returned DataFrames in micromoles per liter!')

    nuts_df = nuts_coords(nuts_df)
    MC_tops_df = MC_tops_coords(MC_coretop_df)

    surface_nuts = nuts_df[nuts_df['Station Type']=='Fish Surface'] 
    

    return nuts_df, surface_nuts, MC_coretop_df, QC_df, reference_df, RMNS_df, LOD_df

def load_Subramaniam_nuts():
    '''
    Loads spreadsheet from Subramaniam et al., 2007 which contains various nutrients and other measurements
    first loads column names that provide compatibility with the MUDBENCS_map function, and then loads
    data, ignoring the statistics from the three different groups of station types in the spreadsheet.
    
    '''
    col_names = [
        'Date',
        'Station', 
        'Station_Type', 
        'Lat Dec. Deg.',
        'Lon Dec. Deg.', 
        'Temperature (C)',
        'Salinity', 
        'Fe_d (nM)',
        'SRP (nM)',
        'Si dissolved (uM)',
        'Si_bio_surf (umol/L)',
        'Nitrate (uM)',
        'DIC (umol/kg), riv corr',
        'Biological drawdown (umol/kg)',
        'Wind speed (m/s)',
        '1 percent depth',
        'MLD (m)',
        'Int Biomass (trichomes/m2 x 10^6)',
        'Int Biomass (richelia/m2 x 10^6)',
        'HPLC Phyto Chl (mg/m2)',
        'Total C fixation (mmolC/m2/day)',
        'Tricho N2 fix (umolN/m2/day)',
        'Hemi N2 fix (umolN/m2/day)', 
        'Total N2 fix (umolN/m2/day)'
    ]

    #Read Subramaniam et al 2007 excel sheet, dropping any column with nans to get rid of the columns with statistics.
    df_1 = pd.read_excel('Subramaniam et al 2007 PNAS Data.xlsx', skiprows=9, usecols='A:X', names=col_names, engine='openpyxl')
    df_1 = df_1.loc[~df_1['Station'].isna()]

    return df_1
