import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

'''
This file contains code to plot Phi as a function of system states
'''


####################################################################################

for ii in np.arange(0,6):
    ID = ii #<< look at which players are selected in which sample

## Load Data
    DF = np.load("./Data/Data_Phi_Test.npy",allow_pickle=True).item()
    team_1_phi = DF['Team_1_Phi']
    team1_time_series_data = DF['Team_1_ts']

# select sample to plot (uses "ID" variable to select which player sets)
    TPM1 = team_1_phi[ID]['TPM'] 



####################################################################################
## Plot distrubtion of Phi over states for both teams
    team1_phi_values = team_1_phi[ID]['phi_values']
    team1_phi_states = team_1_phi[ID]['phi_states']
    team1_max_phi = team_1_phi[ID]['max_phi']
    team1_max_phi_state = team_1_phi[ID]['max_phi_state']
    team1_phi_mechanisms = team_1_phi[ID]['phi_mechanisms']

# Convert states to symbolic representations for labeling
    state_labels = [''.join('●' if bit == 1 else '○' for bit in state) for state in team1_phi_states]

# Convert max_phi_state to the same symbolic representation
    team1_max_phi_state_label = ''.join('●' if bit == 1 else '○' for bit in team1_max_phi_state)

# Define pastel colors
    pastel_color_team1 = '#ADD8E6'  # Light blue for Team 1
    pastel_color_team2 = '#98FB98'  # Pastel green for Team 2
    highlight_color = '#FFC0CB'  # Pastel red for highlighting max phi

# Creating the plot with two subplots
    fig, axs = plt.subplots(1, 1, figsize=(12, 5))

# Plot for Team 1
    for state_label, phi_value in zip(state_labels, team1_phi_values):
        color = highlight_color if state_label == team1_max_phi_state_label else pastel_color_team1
        axs.bar(state_label, phi_value, width=0.5, color=color)

# Setting titles and labels for Team 1 plot
    #axs[0].set_ylim([3.5,100])
    axs.set_title('Team 1 Phi Values - Highlighting Max Phi State')
    axs.set_ylabel('Phi Values',fontsize=15)
    axs.set_xticks(np.arange(len(state_labels)))
    axs.set_xticklabels(state_labels, rotation=90)

# Adjust layout
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.tight_layout()

# Show the plot
    plt.savefig("./Figs/Fig_3_phi_states_"+str(ii)+".png",dpi=300)


