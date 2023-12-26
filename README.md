## Scripts for estimating EFM balloon package descent path

Note: Fall speed function was fit to pressure from an old sounding and may need to be adjusted for changes to the balloon train

Ready to go options:
- RAP model forecast sounding (useful for estimating future sounding bounds): from_rap.py
- Pre-existing environmental sounding (likely highest-resolution wind profile): from_env_sounding.py
- In-progress cutdown-letdown tracker (wind profile estimated from previous movement): from_flight_tracker.py

Requires a 'terrain.txt' file for determining ground level above MSL. Example here was downloaded from topex.ucsd.edu/cgi-bin/get_data.cgi for the Tug Hill Plateau region.

Usage examples are included in the header of each python script. Each script can be called with arguments or as queried user-entered input. For example:
```
python from_rap.py
Enter forecast date and time (YYYYMMDDHH): 2023122717
Enter starting location (lat, lon, alt in m MSL): 43.7, -77.0, 300
Enter expected time from launch to cutdown (minutes): 50
```
AND
```
python from_rap.py 2023122717 43.7 -77.0 300 3000 
# (Note input time here is in seconds)
```
Will each pull the nearest forecast sounding from the RAP for 27 Dec 2023 1700 UTC and estimate a balloon trajectory starting at 43.7 deg N, 77.0 deg W, 300 m MSL which ascends for 50 minutes (3000 seconds) then descends untill it reaches ground level as defined by the terrain.txt file.

Example tracking and environmental soundings also included.

Python package dependencies vary with method and may include: pandas, datetime, numpy, matplotlib, pyproj, xml, requests, urllib
