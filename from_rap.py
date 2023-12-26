# Predict balloon trajectory from a current model sounding and
# specified latitude, longitude, altitude, and time
# to balloon cutdown
#
# Known values are expected as either system arguments during call to script
# with the order: forecast time (YYYYMMDDHH), latitude, longitude, 
# altitude (m/s MSL), time_to_cutdown (s) or queried inputs
#
# Nearest sounding is pulled from https://rucsoundings.noaa.gov/
#
# Example:
# python from_rap.py
# Enter forecast date and time (YYYYMMDDHH): 2023122717
# Enter starting location (lat, lon, alt in m MSL): 43.7, -76.0, 300
# Enter expected time from launch to cutdown (minutes): 20
#
# OR
# python from_rap.py 2023122620 43.7 -77.0 500 3000 
# Note input time in seconds
#
##
# Note: terrain check used to determine if the package has landed
# is only valid over the region of the terrain.txt file
#
# Note: fall function may need to be adjusted following collections
# in representative environment 
# (currently pressure based for consistent tracker estimations)
#
# Writen by Vanna Chmielewski, NOAA/OAR/NSSL
# Last edit: 26 Dec 2023

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pyproj as proj4
import sys

from coordinateSystems import TangentPlaneCartesianSystem, GeographicSystem
geo  = GeographicSystem()

# acquired from topex.ucsd.edu/cgi-bin/get_data.cgi for Tug Hill area
ground = pd.read_csv('terrain.txt', 
            delim_whitespace=True,names=['lon','lat','alt']) # alt in meters
ground['lon'] = ground['lon']-360 # Just for my own comfort

# Defaults and conversions
knts_per_ms = 1.9484
rise_rate = 6.09
show_plot = 'y'
def fall_function(pressure):
    return -917.02/(pressure+11)-5.167

try:
    # Take values from command line if given
    test_dt, balloon_lat, balloon_lon, balloon_alt, time_to_cutdown = [float(x) for x in sys.argv[1:]]
except ValueError:
    # Use one of the two options below - queried input from terminal or command line arguments
    test_dt = input("Enter forecast date and time (YYYYMMDDHH): ")
    balloon_lat, balloon_lon, balloon_alt = [float(x) for x in input('Enter starting location (lat, lon, alt): ').split(',')]
    time_to_cutdown = float(input('Enter expected time from launch to cutdown (minutes): '))*60
    show_plot = input('Show plot of predicted track? (y or n): ')

tanp = TangentPlaneCartesianSystem(balloon_lat, balloon_lon, 0)
test_dt = dt.datetime.strptime(str(int(test_dt)), '%Y%m%d%H')
then = dt.datetime(1970,1,1)

# Find and read sounding from ruc soundings.noaa.gov converting to standard units
url_address = 'https://rucsoundings.noaa.gov/get_soundings.cgi?data_source=Op40&'\
            +test_dt.strftime('start_year=%Y&start_month_name=%b&start_mday=%d&start_hour=%H&start_min=0')\
            +'&n_hrs=1.0&fcst_len=shortest&'\
            +'airport='+str(balloon_lat)+'%2C'+str(balloon_lon)+'&text=Ascii&startSecs='\
            +str(int((test_dt - then).total_seconds())-3600)+'&endSecs='\
            +str(int((test_dt - then).total_seconds()))

sounding =pd.read_csv(url_address, delim_whitespace=True, skiprows=8,
           names = ['pressure (mb)','height (m)','temp (C)',
                    'dewpt (C)','wind dir (deg)','wind spd (kts)'])
sounding['pressure (mb)'] = sounding['pressure (mb)']/10
sounding['dewpt (C)'] = sounding['dewpt (C)']/10
sounding['temp (C)' ] = sounding['temp (C)' ]/10
sounding['u (m/s)'] = -np.sin(sounding['wind dir (deg)']*np.pi/180)*sounding['wind spd (kts)'
                                                                            ]/knts_per_ms
sounding['v (m/s)'] = -np.cos(sounding['wind dir (deg)']*np.pi/180)*sounding['wind spd (kts)'
                                                                            ]/knts_per_ms
# Find fall velocities of profile
sounding['fall_rate'] = -fall_function(sounding['pressure (mb)'])
sounding = sounding.reset_index()

# Interval for ascent estimation and lists for tracking location
interval  = 30 # s
counter   = 0
track_lon = []
track_lat = []
track_alt = []
track_t   = []

# Predict ascent
while counter<=time_to_cutdown:
    # Pull the nearest 2 readings to the balloon altitude (~500 m)
    selection = (sounding['height (m)'] - balloon_alt).abs().nsmallest(2).index
    
    # Advect balloon using mean u,v within level
    delta_x = interval*sounding['u (m/s)'].iloc[selection].mean()
    delta_y = interval*sounding['v (m/s)'].iloc[selection].mean()
    delta_z = interval*rise_rate
     
    # Where is the balloon in local x,y,z?
    balloon_ecef  = np.array(geo.toECEF(balloon_lon, balloon_lat, balloon_alt)).T
    balloon_local = tanp.toLocal(balloon_ecef[:,np.newaxis]).T

    # How much did it move?
    balloon_ecef = tanp.fromLocal(np.array([[balloon_local[0]+delta_x,0], 
                                            [balloon_local[1]+delta_y,0], 
                                            [balloon_local[2]+delta_z,0]]))[:,0]

    # Where is the projected location in lat,lon,alt?
    balloon_lon, balloon_lat, balloon_alt = geo.fromECEF(balloon_ecef[0],
                                                         balloon_ecef[1],
                                                         balloon_ecef[2])

    # Keep track of everything
    track_lon = track_lon + [balloon_lon]
    track_lat = track_lat + [balloon_lat]
    track_alt = track_alt + [balloon_alt]
    track_t = track_t + [counter]
    counter = counter+interval

# Interval for descent estimation -- in distance not time intervals
interval = 50 # m
landed  = False

# Predict descent
while ~landed:
    # Pull the nearest 2 readings to the balloon altitude (~500 m)
    selection = (sounding['height (m)'] - balloon_alt).abs().nsmallest(2).index

    # Advect balloon using mean u,v within level
    delta_x = (interval/sounding.fall_rate.iloc[selection].mean())*sounding['u (m/s)'].iloc[selection].mean()
    delta_y = (interval/sounding.fall_rate.iloc[selection].mean())*sounding['v (m/s)'].iloc[selection].mean()
    
    # Where is the balloon in local x,y,z?
    balloon_ecef  = np.array(geo.toECEF(balloon_lon, balloon_lat, balloon_alt)).T
    balloon_local = tanp.toLocal(balloon_ecef[:,np.newaxis]).T

    # How much did it move?
    balloon_ecef = tanp.fromLocal(np.array([[balloon_local[0]+delta_x,0], 
                                            [balloon_local[1]+delta_y,0], 
                                            [balloon_local[2]-interval,0]]))[:,0]

    # Where is the projected location in lat,lon,alt?
    balloon_lon, balloon_lat, balloon_alt = geo.fromECEF(balloon_ecef[0],balloon_ecef[1],balloon_ecef[2])

    track_lon = track_lon + [balloon_lon]
    track_lat = track_lat + [balloon_lat]
    track_alt = track_alt + [balloon_alt]
    track_t = track_t + [track_t[-1]+interval/sounding.fall_rate[selection].mean()]
    
    # Is that above our ground level dataset?
    landed = (ground.alt[(ground.lat==ground.lat[(ground.lat - balloon_lat).abs().idxmin(skipna=True)])&
                         (ground.lon==ground.lon[(ground.lon - balloon_lon).abs().idxmin(skipna=True)])] 
              > balloon_alt).values[0]


if show_plot == 'y':
    plt.scatter(ground.lon, ground.lat, c=ground.alt, marker='s')
    plt.clim(0,700)
    plt.colorbar(label='Ground altitude')

    plt.scatter(track_lon,track_lat,c=track_alt,cmap='magma')
    plt.colorbar(label='Balloon altitude')
    plt.show()

print ('Estimated landing lat, lon, alt (m MSL): ', "%.4f" % track_lat[-1], "%.4f" % track_lon[-1], "%.1f" % track_alt[-1])
