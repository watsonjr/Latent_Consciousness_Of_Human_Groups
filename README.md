
# INFORMATION INTEGRATION AND THE LATENT CONSCIOUSNESS OF HUMAN GROUPS
## Github repo
This code repo contains a set of python functions used in the paper:

INFORMATION INTEGRATION AND THE LATENT CONSCIOUSNESS OF HUMAN GROUPS
James R. Watson, Alex Maier, Álvaro Novillo, Ignacio Echegoyen, Ricardo Resta, Roberto López del Campo, Javier M. Buldú

contact: James Watson at watsjame@oregonstate.edu for more information

## Summary
In this repo are the functions required to compute Phi (the measure of information integration from the Integrated Information Theory of Conciousness) for soccer player position data. All analysis is conducted in Python, and the list of required packages are provided below. Our code is run in named order:

### 01_make_data.py
This file contains code that loads up the soccer player position data and computes player speeds. 

### 02_make_phi.py
This file contains several functions used in computing timeseries of Phi from soccer speed timeseries, including a function to compute the Transition Probability Matrix (TPM) from the speed timeseries data (compute_transition_probability_matrix), a function to compute Phi from the TPM (calculate_phi_for_tpm), a function to parse out substitutes (parse_subs), and three functions (make_act_1,2 and 3) for computing "activity" from speeds.

As part of the application of these functions, we loop through player subsets (e.g., player pairs, trios, quartets...etc). These are defined by the parameter TPM_n.

Data are stored in the data folder under Data_Phi_Test.npy

### 03_plot_fig1.py
This file contains code to plot timeseries of player speeds and activity

### 04_plot_fig2.py
This file contains code to plot heatmaps for the Transition Probability Matrices 

### 05_plot_fig3.py
This file contains code to plot Phi as a function of system states

### 06_plot_fig4.py
This file contains code to plot Phi over time for player subsets.Note that the data we use here in this repo differ from the data used for the analysis presented in the manuscript. The data provided here does not have the same labels as those of the La Liga data, notably labels for when the soccer ball was in play or not. Consequently, the figures generated here do not look the same as those created for the manuscript. There are frequent and long periods of time where Phi is constant for player subsets. This feature should be ignored and are attributed to this difference between the datasets. In the analysis of the La Liga data, these features were not found.

### Requirements
Make sure you have this version of IIT installed:

!pip install -U git+https://github.com/wmayner/pyphi.git@feature/iit-4.0

We also use the following python Packages:
- numpy
- pandas
- matplotlib
- itertools
- time
- random
- seaborn


### Data Provided in this Repo
Soccer player position data in the manuscript come from the Spanish National League (La Liga) during the 2018/2019 season. However, we are not allowed to share these data. Consequently, for this repo we provide equivalent data found here:

https://github.com/metrica-sports/sample-data

Here we use one of their sample datasets in:

/sample-data/data/Sample_Game_1/

Specifically

https://github.com/metrica-sports/sample-data/blob/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv

We provide this csv file in the Data folder of this repo. We have manually extracted the relevent data so that the subsequent functions can be run.  



