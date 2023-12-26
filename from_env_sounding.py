# Predict balloon trajectory from an existing sounding and
# known latitude, longitude, altitude, fixed rise rate and time
# to balloon cutdown
#
# Known values are expected as either system arguments during call to script
# with the order: latitude, longitude, 
# altitude (m/s MSL), time_to_cutdown (s) or queried inputs
#
# Example:
# python from_env_sounding.py 43.7 -77.0 500 3000 
# Note input time in seconds
#
# OR
# python from_env_sounding.py
# Enter starting location (lat, lon, alt in m MSL): 43.7, -76.0, 300
# Enter expected time from launch to cutdown (minutes): 20
#
# Adjust sounding block for given data format
#
# Note: terrain check used to determine if the package has landed
# is only valid over the region of the terrain.txt file
#
# Note: fall function may need to be adjusted following collections
# in representative environment 
# (currently pressure based as previous density obs were suspect)
#
# Writen by Vanna Chmielewski, NOAA/OAR/NSSL
# Last edit: 26 Dec 2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyproj as proj4
import sys

from coordinateSystems import TangentPlaneCartesianSystem, GeographicSystem
geo  = GeographicSystem()

ground = pd.read_csv('terrain.txt', 
            delim_whitespace=True,names=['lon','lat','alt']) # alt in meters
# acquired from topex.ucsd.edu/cgi-bin/get_data.cgi
ground['lon'] = ground['lon']-360 # Just for my comfort

show_plot ='n'
rise_rate = 6.09
def fall_function(pressure):
    return -917.02/(pressure+11)-5.167

try:
    # # Take balloon location input, set map at same location for calculations from command-line call
    balloon_lat, balloon_lon, balloon_alt, time_to_cutdown = [float(x) for x in sys.argv[1:]]
except ValueError:
    # Or inputs if not given
    balloon_lat, balloon_lon, balloon_alt = [float(x) for x in input('Enter starting location (lat, lon, alt): ').split(',')]
    time_to_cutdown = float(input('Enter expected time from launch to cutdown (minutes): '))*60
    show_plot = input('Show plot of predicted track? (y or n): ')

tanp = TangentPlaneCartesianSystem(balloon_lat, balloon_lon, 0)


##############
# Read sounding, adjust parameters based on sounding file format
# # Type 1
# sounding = pd.read_csv('SD13-2014_2007-01-3_080214.bin.asci.Exyz.csv',
#            skiprows=[0,1,2,4], na_values=-999999)
# sounding.columns = sounding.columns.str.replace(' ', '')
# gps_available = 'yes'
# # Any renamings go here
# sounding.rename(columns={'Lat':'lat',
#                          'Lon':'lon',
#                          'Alt(m)':'alt'},inplace=True)

# Type 2 - no lat/lon, assume it's at the input location
sounding = pd.read_csv('OSW20220901_02Z_DigiCora.txt',
           delim_whitespace=True, skiprows=1,
           names=['Time','alt','Pres(mb)','temp','rh','dew','Wdir(deg)','Wspd(m/s)'],
                        na_values=['/////','EOF'])
sounding = sounding.astype(float, errors='ignore')
sounding['lon'], sounding['lat'] = balloon_lon, balloon_lat
gps_available = 'no'
###############

# Convert wind direction + speed to u + v
sounding['u'] = -np.sin(sounding['Wdir(deg)']*np.pi/180)*sounding['Wspd(m/s)']
sounding['v'] = -np.cos(sounding['Wdir(deg)']*np.pi/180)*sounding['Wspd(m/s)']
# Find fall velocities of profile
sounding['fall_rate'] = -fall_function(sounding['Pres(mb)'])

# Subset if necessary
sounding = sounding[sounding.index<np.argmax(sounding['alt'])]

#######
# # Alternatively, pull balloon location from sounding
# balloon_lon = sounding.iloc[sounding['lon'].last_valid_index()]['lon']
# balloon_lat = sounding.iloc[sounding['lat'].last_valid_index()]['lat']
# balloon_alt = sounding.iloc[sounding['alt'].last_valid_index()]['alt']
# time_to_cutdown=4063
########

# interval for rise estimations (time)
interval = 10 # s
counter = 0

track_lon = []
track_lat = []
track_alt = []
track_t = []

# Predict ascent
while counter<=time_to_cutdown:
    # Pull the nearest 10 readings to the balloon altitude
    # With a 5 m/s rise rate and 1 Hz data, ~100 m of data
    selection = (sounding.alt - balloon_alt).abs().nsmallest(10).index
    
    # Advect balloon using mean u,v within level
    delta_x = interval*sounding.u.iloc[selection].mean()
    delta_y = interval*sounding.v.iloc[selection].mean()
    delta_z = interval*rise_rate
    
    if (np.abs(sounding.alt - balloon_alt).min()>100) & (gps_available == 'yes'):
        # If there is a gap in data of at least 100 m and there is GPS, look for 
        # points above and below the balloon altitude to estimate wind speeds instead
        # Only applies if sounding data exists above the altitude (not the max, or min)
        sel1 = (sounding.alt - balloon_alt)[sounding.alt - balloon_alt<0].nlargest(10).index
        sel2 = (sounding.alt - balloon_alt)[sounding.alt - balloon_alt>0].nsmallest(10).index

        if (len(sel1)>0) & (len(sel2)>0):
            lat1,lon1 = sounding.lat.iloc[sel1].mean(), sounding.lon.iloc[sel1].mean(), 
            alt1,t1   = sounding.alt.iloc[sel1].mean(), sounding.Time.iloc[sel1].mean()
            lat2,lon2 = sounding.lat.iloc[sel2].mean(), sounding.lon.iloc[sel2].mean(), 
            alt2,t2   = sounding.alt.iloc[sel2].mean(), sounding.Time.iloc[sel2].mean()

            gap_ecef  = np.array(geo.toECEF(np.array([lon1,lon2]), np.array([lat1,lat2]), np.array([alt1,alt2]))).T
            gap_local = tanp.toLocal(gap_ecef.T).T

            movement = gap_local[1]-gap_local[0]
            delta_x = movement[0]/(t2-t1)
            delta_y = movement[1]/(t2-t1)
     
    # Where is the balloon in local x,y,z?
    balloon_ecef  = np.array(geo.toECEF(balloon_lon, balloon_lat, balloon_alt)).T
    balloon_local = tanp.toLocal(balloon_ecef[:,np.newaxis]).T

    # How much did it move?
    balloon_ecef = tanp.fromLocal(np.array([[balloon_local[0]+delta_x,0], 
                                            [balloon_local[1]+delta_y,0], 
                                            [balloon_local[2]+delta_z,0]]))[:,0]

    # Where is the projected location in lat,lon,alt?
    balloon_lon, balloon_lat, balloon_alt = geo.fromECEF(balloon_ecef[0],balloon_ecef[1],balloon_ecef[2])

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
    # Pull the nearest 10 readings to the balloon altitude
    # With a 5 m/s rise rate and 1 Hz data, ~100 m of data
    selection = (sounding.alt - balloon_alt).abs().nsmallest(10).index

    # Advect balloon using mean u,v within level
    delta_x = (interval/sounding.fall_rate.iloc[selection].mean())*sounding.u.iloc[selection].mean()
    delta_y = (interval/sounding.fall_rate.iloc[selection].mean())*sounding.v.iloc[selection].mean()

    if (np.abs(sounding.alt - balloon_alt).min()>100) & (gps_available == 'yes'):
        # If there is a gap in data of at least 100 m, look for 
        # points above and below the balloon altitude to estimate wind speeds instead
        # Only applies if sounding data exists above the altitude (not the max, or min)
        sel1 = (sounding.alt - balloon_alt)[sounding.alt - balloon_alt<0].nlargest(10).index
        sel2 = (sounding.alt - balloon_alt)[sounding.alt - balloon_alt>0].nsmallest(10).index

        if (len(sel1)>0) & (len(sel2)>0):
            lat1,lon1 = sounding.lat.iloc[sel1].mean(), sounding.lon.iloc[sel1].mean(), 
            alt1,t1 = sounding.alt.iloc[sel1].mean(), sounding.Time.iloc[sel1].mean()
            lat2,lon2 = sounding.lat.iloc[sel2].mean(), sounding.lon.iloc[sel2].mean(), 
            alt2,t2 = sounding.alt.iloc[sel2].mean(), sounding.Time.iloc[sel2].mean()

            gap_ecef  = np.array(geo.toECEF(np.array([lon1,lon2]), np.array([lat1,lat2]), np.array([alt1,alt2]))).T
            gap_local = tanp.toLocal(gap_ecef.T).T

            movement = gap_local[1]-gap_local[0]
            delta_x = movement[0]/(t2-t1)
            delta_y = movement[1]/(t2-t1)
    
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

# print (track_t[-1]/60, ' minutes to impact')

print ('Final longitude: ', track_lon[-1], 
       'Final latitude: ', track_lat[-1], 
       'Final altitude: ', track_alt[-1], 
       'Minutes until landing: ', track_t[-1]/60)

if show_plot=='y':
    plt.scatter(ground.lon, ground.lat, c=ground.alt, marker='s')
    plt.clim(0,700)
    plt.colorbar(label='Ground altitude')
    # plt.scatter(balloon_lon, balloon_lat)
    plt.scatter(track_lon, track_lat,s=1)
    # plt.ylim(43.3,44)
    # plt.xlim(-76.1, -75.25)
    plt.ylim(np.min(track_lat)-0.1, np.max(track_lat)+0.1)
    plt.xlim(np.min(track_lon)-0.1, np.max(track_lon)+0.1)
    plt.show()