import numpy as np
import pandas as pd
import pyphi
import matplotlib.pylab as plt
from itertools import product
import itertools
import time
from random import randint
import random
from itertools import combinations

'''
Code to compute Transition Probability Matrixes (TPMs) and Phi from soccer player speed timeseries 
'''

# Don't forget you need to install latest version of python and pyphi:
# !pip install -U git+https://github.com/wmayner/pyphi.git@feature/iit-4.0


# Disable PyPhi progress bars and welcome message
pyphi.config.PROGRESS_BARS = False
pyphi.config.WELCOME_OFF = True

##########################################################################################################################################################################################
'''
Functions
'''
## Function for creating non-random sets of players to analyze Phi with
def generate_sequences(n, total_sequences):
    return [list(range(i, i + n)) for i in range(1, total_sequences + 1)]

## Function to compute Transition Probability Matrices (TPMs)
def compute_transition_probability_matrix(time_series_data):
    # Infer the number of states from the data (assuming binary data, the max value + 1)
    num_states = int(np.max(time_series_data) + 1)

    # Number of players
    num_players = time_series_data.shape[0]

    # Total number of possible states (2^num_players)
    total_states = num_states ** num_players

    # Initialize the transition probability matrix with zeros
    TPM = np.zeros((total_states, total_states))

    # Dynamically generate the correct order of states
    all_states = list(product(range(num_states), repeat=num_players))
    correct_order = sorted(all_states, key=lambda x: (sum(x), x[::-1]))
    state_to_index = {state: index for index, state in enumerate(correct_order)}

    # Count the transitions
    for t in range(time_series_data.shape[1] - 1):
        current_state = tuple(time_series_data[:, t])
        next_state = tuple(time_series_data[:, t + 1])
        current_index = state_to_index[current_state]
        next_index = state_to_index[next_state]
        TPM[current_index, next_index] += 1

    # Normalize the rows to get probabilities
    for i in range(total_states):
        row_sum = np.sum(TPM[i, :])
        if row_sum > 0:
            TPM[i, :] /= row_sum

    return TPM, correct_order

# Function to calculate phi values and states for a single TPM, tracking skipped trials
def calculate_phi_for_tpm(tpm, possible_states, labels):
  converted_tpm = pyphi.convert.sbs2sbn(tpm)
  network = pyphi.Network(converted_tpm, node_labels=labels)
  node_indices = list(range(0, len(labels)))

  # Check if possible_states is a single state and convert it to a list of states if necessary
  if isinstance(possible_states[0], int):
    possible_states = [possible_states]

  phi_values = np.zeros(len(possible_states))
  phi_states = np.zeros((len(possible_states), len(labels)), dtype=int)

  max_phi = -np.inf
  max_phi_state = None
  phi_mechanisms = []

  for i, state in enumerate(possible_states):
        candidate_system = pyphi.Subsystem(network, state, node_indices)
        phi_structure = pyphi.new_big_phi.phi_structure(candidate_system)
        big_phi = phi_structure.big_phi
        phi_values[i] = big_phi
        phi_states[i] = state
        current_state_mechanisms = [distinction.mechanism for distinction in phi_structure.distinctions]
        phi_mechanisms.append(current_state_mechanisms)
        if big_phi > max_phi:
            max_phi = big_phi
            max_phi_state = state

  return phi_values, phi_states, max_phi, max_phi_state, phi_mechanisms

# Function to determine the highest order of mechanism
def highest_order(mechanisms):
    return max(len(m) for m in mechanisms) if mechanisms else 0

# Function to find subs and parse them out
def parse_subs(ARR):
    # find subs
    ID = np.where(np.isnan(ARR[:,0])==1)[0]
    JD = np.where(np.isnan(ARR[:,0])==0)[0]

    SUB = ARR[ID,:] # parse out subs
    ARR = ARR[JD,:] # parse out mains

    N = ARR.shape[0] # number of main players
    S = SUB.shape[0] # number of subs
    M = ARR.shape[1] # number of timesteps
    ln = np.zeros(N)
    sn = np.zeros(S)

    # find mains that were subbed out
    for i in np.arange(0,N):
        sp = ARR[i,:]
        ln[i] = len(np.where(np.isnan(sp))[0])
    R = len(np.where(ln)[0]) # number of main players that came off (incl red cards)

    # find subs
    for i in np.arange(0,S):
        sn[i] = M - len(np.where(np.isnan(SUB[i,:]))[0])

    # add noise (if subs have identical sub-in times)a
    i = np.where(ln)[0]
    rn = np.arange(1,R+1)
    random.shuffle(rn)
    ln[i] = ln[i] + rn

    rn = np.arange(1,S+1)
    random.shuffle(rn)
    sn = sn + rn

    # match subs with main players that came off
    i = np.where(ln)[0] # index of main players that were substituted
    j = np.zeros(len(i)) # index of subs that came on
    for k in np.arange(0,len(i)):
        m = ln[i[k]]
        #j[k] = np.where(sn == m)[0][0] # old way (if index is exact)
        d = np.abs(sn-m)
        if min(d) > 200:
            j[k] = np.nan
        else:
            j[k] = np.argmin(np.abs(sn-m)) # new way
    ID_subs = np.vstack((i,j)).transpose() # index to save (col1 = main players, col2 = subs that came in)

    # If there was someone sent off
    i = np.where(np.isnan(ID_subs[:,1]))[0]
    if len(i) != 0:
        KD = ID_subs[i,0] # index of players sent off (to remove at end)
        ID_subs = np.delete(ID_subs,i,0)

        # Remove/zero players that got a red card
        # ARR = np.delete(ARR,KD,0) # if you just want to delete player
        for i in np.arange(0,len(KD)):
            j = np.where(np.isnan(ARR[int(KD[i]),:]))[0]
            ARR[int(KD[i]),j] = 0.

    # Make consistent timeseries array
    for i in np.arange(0,ID_subs.shape[0]):
        j = int(ID_subs[i,0])
        k = int(ID_subs[i,1])
        A = ARR[j,:]
        B = SUB[k,:]
        A[np.isnan(A)] = 0
        B[np.isnan(B)] = 0
        ARR[j,:] = A + B

    # Find goalie
    OUT = ARR
    Sp = np.mean(OUT,1)
    ID_goalie = np.argmin(Sp)
    OUT = np.delete(OUT,ID_goalie,0)

    return OUT

# Function to make activation from speeds (faster than others)
def make_act_1(ARR):
    T = ARR.shape[1]
    OUT = np.zeros(ARR.shape)
    for i in np.arange(0,T):
        ln = ARR[:,i]
        th = np.nanpercentile(ln,50) # choose how sensitive speed threshold
        out = np.zeros(ln.shape)
        out[ln>=th] = 1
        out[np.where(np.isnan(ln))]=np.nan
        OUT[:,i] = out
    return OUT

# Function to make activation from acceleration
def make_act_2(ARR):
    out = np.diff(ARR,1)
    OUT = np.zeros(out.shape)
    OUT[np.where(out>=0)] = 1
    zero_column = np.zeros((out.shape[0], 1))
    OUT = np.hstack((zero_column,OUT))
    ID = np.where(np.isnan(ARR[:,0]))[0]
    OUT[ID,:] = np.nan
    return OUT

# Function to make activation from speeds (faster than your mean)
def make_act_3(ARR):
    N = ARR.shape[0]
    OUT = np.zeros(ARR.shape)
    for i in np.arange(0,N):
        ln = ARR[i,:]
        th = np.mean(ln)
        if np.isnan(th):
            OUT[i,:] = np.nan
        else:
            #th = np.percentile(ln,60) # choose how sensitive speed threshold
            ID = np.where(ln>=th)
            OUT[i,ID] = 1
    return OUT




##########################################################################################################################################################################################
'''
Goal is to calculate Phi for each pair of competing teams 
In this test we use data from: 
https://github.com/metrica-sports/sample-data
which are different from the Spanish La Liga data we present in the manuscript
but similar enough for our code to work on
'''

### Define the size of the player subset
TPM_n = 2 # 2: duos, 3: trios. 4: quartets (any larger and the code will take too long to run)

# load game data
spd_ar = np.load("./Data/Data_speeds.npz")['arr_0']
spd_ar = spd_ar.transpose() # transpose so its the right shape for analysis
spd_ar = spd_ar[:,3000:20000] # subsample for quick analysis (only for this demo)

# Convert the list of moving averages back into a numpy array
#spd_ar = np.array(moving_averages)
arr = spd_ar

## Parse out subs
ARR_1 = parse_subs(arr) 

## ID red cards (make any player timeseries a nan if they got a red card)
for i in np.arange(0,ARR_1.shape[0]):
    # Find indices where the array is zero
    ts = ARR_1[i,:]
    zero_indices = np.where(ts == 0)[0]
    consecutive_groups = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
    consecutive_lengths = [len(group) for group in consecutive_groups]

    if max(consecutive_lengths) > 100:
        ARR_1[i,:] = np.nan

# define number of time points
T = ARR_1.shape[1]

## Make activation timeseries
ACT_1 = make_act_3(ARR_1)

## sequences of ALL COMBINATIONS of players to compute Phi for
numbers = list(range(ACT_1.shape[0]))  # Numbers 0 to 9
sequences = list(combinations(numbers, TPM_n))
EX = len(sequences)

# dicts to sample samples
Team_1_Phi = [{}] * EX 

# For each random selection of #TPM_n players, and for EX times
for k in np.arange(0,EX):

    ## Non-randomly select players
    ID = sequences[k]

    # get necessary stuff for Phi calculation
    labels = [str(i) for i in ID]
    num_bits = len(labels)
    possible_states = np.array(list(itertools.product([0, 1], repeat=num_bits)))

    # get timeseries for two teams
    time_series_list_1 = np.zeros((TPM_n,T))
    
    for ii in np.arange(0,TPM_n):
        
        # must check if there are red cards (naned out and ignored)
        if  np.isnan(ACT_1[ID[ii],0]):
            ts_1 = ACT_1[ID[ii],:]
        else:
            ts_1 = ACT_1[ID[ii],:].astype(int)

        # no null
        time_series_list_1[ii,:] = ts_1

    # TEAM 1: Compute TPM using Alex's function
    if np.any(np.isnan(time_series_list_1)):
        # if nans (red cards), ignore
        TPM1 = np.nan
        possible_states_1 = np.nan
        phi_results = [[]]*5
        phi_results[0] = np.nan
        phi_results[1] = np.nan
        phi_results[2] = np.nan
        phi_results[3] = np.nan
        phi_results[4] = np.nan
    else:
        # if no nans, no red cards, then compute Phi
        TPM1, possible_states_1 = compute_transition_probability_matrix(time_series_list_1)
        phi_results = calculate_phi_for_tpm(TPM1, possible_states_1, labels)

    # TEAM 1: create a dict to save
    Team_1_Phi[k] = {'Sampled_players': ID, 
          'phi_values': phi_results[0],
          'phi_states': phi_results[1],
          'max_phi': phi_results[2],
          'max_phi_state': phi_results[3],
          'phi_mechanisms': phi_results[4],
          'TPM':TPM1}

# create save dict and save
DF = {'Player_sets':sequences,
      'Possible_states_1':possible_states_1,
      'Team_1_ts':ACT_1,
      'Team_1_ss':ARR_1,
      'Team_1_Phi':Team_1_Phi}
np.save('./Data/Data_Phi_Test', DF)

