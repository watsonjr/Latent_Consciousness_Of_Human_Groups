import numpy as np
import pandas as pd
import matplotlib.pylab as plt

'''
Code to compute soccer player speeds from player position data
'''

## Import player position data
data = pd.read_csv("./Data/Sample_Game_1_RawTrackingData_Away_Team_trimmed.csv")

## Convert x,y coordinates into speed (s = sqrt(dx^2 + dy^2) / dt)
T, N = data.shape # number of time points, number of players
N = N - 1 # account for the time column
#N = int((N - 1)/2) # each player has an x and y column, and we subtract the time column

## Use numpy arrays
data_arr = np.asarray(data)
time     = data_arr[:,0] # array of time (first col)
locs     = data_arr[:,1:] # array of locations (x,y) - all other cols

# compute time difference
dt = np.diff(time)

# make an empty array to fill
SPD = np.zeros((T-1,int(N/2)))

# fill it up with player speeds
i = np.arange(0,N,2)
j = np.arange(1,N,2)

k = 0
for l in np.arange(0,len(i)):
    x = locs[:,i[l]]
    y = locs[:,j[l]]

    dx = np.diff(x)
    dy = np.diff(y)

    s  = np.sqrt(dx**2 + dy**2) / dt

    SPD[:,k] = s
    k+=1

# Save as a numpy datafile to load later
np.savez("./Data/Data_speeds",SPD)



