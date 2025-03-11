import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib



## Load Data
pl = 44 # number of player groups (120 for trios, 44 for duos, 7 for quartets)
DF = np.load("./Data/Data_Phi_Test.npy",allow_pickle=True).item()
team_1_phi = DF['Team_1_Phi']
team1_time_series_data = DF['Team_1_ts']


####################################################################################
OUT = [[]]*pl
for ii in np.arange(0,pl):
    ID = ii #<< look at which players are selected in which sample

# Extracting the necessary data from phi_results
    phi_values = team_1_phi[ID]['phi_values']
    phi_states = team_1_phi[ID]['phi_states']
    i = team_1_phi[ID]['Sampled_players']
    team1_selected_players_data = team1_time_series_data[i,:]

# Mapping each state to its corresponding phi value
    if np.any(np.isnan(team1_selected_players_data)):
        OUT[ii] = np.nan

    else:
        state_to_phi = {tuple(state): phi for state, phi in zip(phi_states, phi_values)}

        # Reassigning the phi values to the respective states in the original time series sequence
        phi_time_series = np.array([state_to_phi[tuple(state)] for state in team1_selected_players_data.T])

        # Save 
        OUT[ii] = phi_time_series

###### Plotting the time series of PHI values and the moving average
# Set up the figure and axes
OUT = [sublist for sublist in OUT if not np.isnan(sublist).any()]
fig, ax = plt.subplots(figsize=(12, 4))

# Number of lines
num_lines = len(OUT)

# Define the colormap
base_cmap = matplotlib.colormaps['coolwarm']
color_list = base_cmap(np.linspace(0, 1, num_lines))
cmap = mcolors.ListedColormap(color_list)
#custom_tick_labels = ['Defense', '...', '...', 'Midfield', '...','...','Forwards']

# Example data
x = np.arange(0,OUT[0].shape[0])
y_lines = OUT

# Plotting the lines
for i, y in enumerate(y_lines):
    ax.plot(x, y, color=cmap(i))
    #ax.set_ylim(0, 100)  # Replace 'ymin' and 'ymax' with your desired limits

# Plot mean and max
mn = np.mean(np.asarray(y_lines),0)
vr = np.var(np.asarray(y_lines),0)
mx = np.max(np.asarray(y_lines),0)
ax.plot(x,mn,color='k')
ax.plot(x,mx,color='k')

# Create a colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, num_lines - 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=range(num_lines))
cbar.set_label('Player set index')
#cbar.set_ticklabels(custom_tick_labels)

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title("Team 1 Phi (moving ave) for different players sets")
plt.xlabel("Time")
plt.ylabel("Phi")
plt.savefig("./Figs/Fig_4_phi_team1_ts_playersets_"+str(ii)+".png",dpi=300)


