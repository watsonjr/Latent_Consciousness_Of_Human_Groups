import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

'''
This file contains code to plot heatmaps for the Transition Probability Matrices
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


## TPMs
# Function to plot a heatmap with adjusted color scale limits
    plt.close('all')
    def plot_adjusted_heatmap(TPM, title, color_limit):
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(TPM, cmap='viridis', vmin=0, vmax=color_limit)
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.tight_layout()

# Determining an appropriate color scale limit
# Ignoring the diagonal elements for this calculation
    diagonal_ignored_TPM1 = np.copy(TPM1)
    np.fill_diagonal(diagonal_ignored_TPM1, 0)

# Find the maximum value in the off-diagonal elements to set as the color limit
    color_limit1 = np.max(diagonal_ignored_TPM1)

# Plotting adjusted heatmaps for both Team 1 and Team 2
    plot_adjusted_heatmap(TPM1, 'Adjusted Heatmap of TPM for Team 1', color_limit1)
    plt.savefig('./Figs/Fig_2_tpm_team1_'+str(ID)+'.png',dpi=300)




