import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

'''
This file contains code to plot timeseries of player speeds and activity
'''

######################################################################################################################################################################
## Load data
DF = np.load("./Data/Data_Phi_Test.npy",allow_pickle=True).item()
ARR_1 = DF['Team_1_ss']
ACT_1 = DF['Team_1_ts']

## PLOT timeseries with subs (speed)
ts = ARR_1
plt.figure(figsize=(8, 4))
ax = sns.heatmap(ts, cmap="YlGnBu", cbar_kws={'label': 'Speed'}) # activation
plt.xlabel("Time",fontsize=15)
plt.ylabel("Player ID",fontsize=15)
for spine in ax.spines.values():
    spine.set_visible(True)     # Make the border visible
    spine.set_linewidth(1.5)    # Set the border width
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.tight_layout()
plt.savefig("./Figs/Fig_1_team_1_speeds_subs.png",dpi=300)

##### PLOT timeseries of just a few players
ts1 = ARR_1
plt.figure(figsize=(8, 4))
plt.plot(ts1[2,:]) # activation
plt.xlabel("Time",fontsize=15)
plt.ylabel("Player ID",fontsize=15)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.tight_layout()
plt.savefig("./Figs/Fig_1_player_1_speeds_ts.png",dpi=300)

plt.figure(figsize=(8, 5))
plt.hist(ts1[3,:],50,alpha=0.75,density=True,color='r') # activation
#plt.xlim([0,30])
#plt.ylim([0,0.15])
plt.xlabel("Speed (miles/hr)",fontsize=15)
plt.ylabel("Freq(Speed)",fontsize=15)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.tight_layout()
plt.savefig("./Figs/Fig_1_player_1_speeds_hist.png",dpi=300)

## Make activation timeseries
ts = ACT_1
plt.figure(figsize=(8, 4))
ax = sns.heatmap(ts, cmap="YlGnBu", cbar_kws={'label': 'Activity'}) # activation
plt.xlabel("Time",fontsize=15)
plt.ylabel("Player ID",fontsize=15)
for spine in ax.spines.values():
    spine.set_visible(True)     # Make the border visible
    spine.set_linewidth(1.5)    # Set the border widthplt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.tight_layout()
plt.savefig("./Figs/Fig_1_team_1_act_nosubs.png",dpi=300)


