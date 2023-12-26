# Predict balloon trajectory from only the tracked path by the letdown-cutdown
#
# Tracker of interest and tracker file can be specified in command
# line arguments or as queried. A tracker file of 'current' will
# pull from the live website
#
# Example:
# python from_flight_tracker.py
# Enter path to cutdown tracker file OR 'current': NSSL1313_cutdown1119.xml
# Enter tracker number: 209825
#
# OR
# python from_flight_tracker.py NSSL1313_cutdown1119.xml 209825
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
import sys
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pyproj as proj4
import urllib
import requests
from xml.dom.minidom import *

from coordinateSystems import TangentPlaneCartesianSystem, GeographicSystem
geo  = GeographicSystem()

ground = pd.read_csv('terrain.txt', 
            delim_whitespace=True,names=['lon','lat','alt']) # alt in meters
# acquired from topex.ucsd.edu/cgi-bin/get_data.cgi for Tug Hill area
ground['lon'] = ground['lon']-360 # Just for my own comfort

show_plot = 'n'
rise_rate = 6.09 # m/s
def fall_function(pressure):
    return -917.02/(pressure+11)-5.167

try:
    # Take values from command line if given
    filepath, tracker_interest = [str(x) for x in sys.argv[1:]]
except ValueError:
# Use one of the two options below - queried input from terminal or command line arguments
    filepath = input("Enter path to cutdown tracker file OR 'current': ")
    tracker_interest = input("Enter tracker number: ")
    show_plot = input('Show plot of predicted track? (y or n): ')

# Read xml file or directly from website
if filepath == 'current':
    res = urllib.request.urlopen('http://kennedy.tw:8001/path/NSSL1313')
    dom = parseString(res.read())
else:
    dom = parse(filepath)

# Parse xml file for locations and pressures for the given tracker
placemarks = dom.getElementsByTagName("Placemark")
for track_count, i in enumerate(placemarks):
    Tracker = i.getElementsByTagName('name')[0].childNodes[0].data
    if Tracker == tracker_interest:
        node = i.getElementsByTagName("gx:Track")
        times = []
        lats = []
        lons = []
        alts = []
        pressure = []
        for count, j in enumerate(node[0].childNodes):
            if count%2==1:
                if j.tagName == 'when':
                    times.append(dt.datetime.strptime(j.firstChild.data,'%Y-%m-%dT%H:%M:%SZ'))
                if j.tagName == 'gx:coord':
                    lats.append(float(j.firstChild.data.split(' ')[1]))
                    lons.append(float(j.firstChild.data.split(' ')[0]))
                    alts.append(float(j.firstChild.data.split(' ')[2]))
        node2 = i.getElementsByTagName("gx:SimpleArrayData")
        for k in range(len(node2)):
            if node2[k].attributes.get('name').value == 'pressure':
                for count, j in enumerate(node2[k].childNodes):
                    if count%2==1:
                        pressure.append(float(j.firstChild.data.split(' ')[0]))

# # If a time check is needed later:
# relative_t = [(dt.datetime.utcnow() - time).seconds for time in times]
# selects = np.array(relative_t)/60<500

# Put into one dataframe
sounding = pd.DataFrame(np.array([np.array(lats), np.array(lons),
                                  np.array(alts), np.array(times),
                                  np.array(pressure)]).T,
                        columns = ['lat','lon','alt','time','pressure'])
sounding['fall_rate'] = -fall_function(sounding['pressure'])

# Assume location of staring descent is the last known location
balloon_lat, balloon_lon, balloon_alt, time_to_cutdown = [sounding.lat.iloc[-1],
                                                          sounding.lon.iloc[-1],
                                                          sounding.alt.iloc[-1],0]

# Put tangent plane for calculations at the ground level of the location
tanp = TangentPlaneCartesianSystem(balloon_lat, balloon_lon, 0)

# Interval for descent estimation and starting lists for tracking values
interval = 50 # m
landed  = False
counter = 0
track_lon = []
track_lat = []
track_alt = []
track_t = [0]

# Estimate future path
while ~landed:
    # Pull the nearest altitudes to the estimated descent path to estimate winds
    sel1 = (sounding.alt-balloon_alt)[sounding.alt-balloon_alt< 0].astype(float).nlargest(1).index
    sel2 = (sounding.alt-balloon_alt)[sounding.alt-balloon_alt>=0].astype(float).nsmallest(1).index
    
    # Find full locations and times at those altitudes (if frequent can use multiple levels above)
    lat1,lon1 = sounding.lat.iloc[sel1].mean(), sounding.lon.iloc[sel1].mean()
    alt1,t1   = sounding.alt.iloc[sel1].mean(), sounding.time.iloc[sel1].mean()
    lat2,lon2 = sounding.lat.iloc[sel2].mean(), sounding.lon.iloc[sel2].mean()
    alt2,t2   = sounding.alt.iloc[sel2].mean(), sounding.time.iloc[sel2].mean()

    # Convert to a local x,y,z
    gap_ecef  = np.array(geo.toECEF(np.array([lon1,lon2]), np.array([lat1,lat2]), np.array([alt1,alt2]))).T
    gap_local = tanp.toLocal(gap_ecef.T).T

    # Determine depth fallen in interval and corresponding advection
    movement = (interval/sounding.fall_rate.iloc[sel1].mean())*(gap_local[1]-gap_local[0])
    delta_x = (movement[0]/(t2-t1).seconds) # in m
    delta_y = (movement[1]/(t2-t1).seconds) # in m

    # Where was the balloon in local x,y,z?
    balloon_ecef  = np.array(geo.toECEF(balloon_lon, balloon_lat, balloon_alt)).T
    balloon_local = tanp.toLocal(balloon_ecef[:,np.newaxis]).T

    # How much did it move?
    balloon_ecef = tanp.fromLocal(np.array([[balloon_local[0]+delta_x,0], 
                                            [balloon_local[1]+delta_y,0], 
                                            [balloon_local[2]-interval,0]]))[:,0]

    # Where is the projected location in lat,lon,alt?
    balloon_lon, balloon_lat, balloon_alt = geo.fromECEF(balloon_ecef[0],
                                                         balloon_ecef[1],
                                                         balloon_ecef[2])

    # Track location along projected descent track
    track_lon = track_lon + [balloon_lon]
    track_lat = track_lat + [balloon_lat]
    track_alt = track_alt + [balloon_alt]
    track_t = track_t + [track_t[-1]+interval/sounding.fall_rate[sel1].mean()]
    
    # Is that above our ground level dataset?
    landed = (ground.alt[(ground.lat==ground.lat[(ground.lat - balloon_lat).abs().idxmin(skipna=True)])&
                         (ground.lon==ground.lon[(ground.lon - balloon_lon).abs().idxmin(skipna=True)])] 
              > balloon_alt).values[0]


print (track_t[-1]/60, ' minutes to impact')
print ('Estimated landing lat, lon, alt (m MSL): ', "%.4f" % track_lat[-1], "%.4f" % track_lon[-1], "%.1f" % track_alt[-1])

if show_plot=='y':
    plt.scatter(ground.lon, ground.lat, c=ground.alt, marker='s')
    plt.clim(0,700)
    plt.colorbar(label='Ground altitude')
    plt.scatter(track_lon, track_lat,s=1)
    plt.ylim(np.min(track_lat)-0.1, np.max(track_lat)+0.1)
    plt.xlim(np.min(track_lon)-0.1, np.max(track_lon)+0.1)
    plt.show()