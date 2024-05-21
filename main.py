import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

# Load the datasets
office_data = pd.read_csv('1. OfficeIndoorClimate.csv')
delhi_data = pd.read_csv('3. DailyDelhiClimate.csv')


# Discretize the continuous data into discrete states
def discretize_data(data, num_bins):
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
    data_reshaped = data.values.reshape(-1, 1)
    data_discretized = discretizer.fit_transform(data_reshaped)
    return data_discretized.astype(int).flatten()


# Discretize the temperature data from both datasets
num_states = 3  # Number of discrete states
office_temp = discretize_data(office_data['ki [C]'], num_states)
delhi_temp = discretize_data(delhi_data['meantemp'], num_states)


# Define a function to fit HMM and compute log likelihood
def fit_hmm(data, num_states):
    model = hmm.CategoricalHMM(n_components=num_states, n_iter=100, )
    data_reshaped = data.reshape(-1, 1)
    model.fit(data_reshaped)
    log_likelihood = model.score(data_reshaped)
    return model, log_likelihood


# Fit HMMs with different numbers of states and store the log likelihoods
state_range = range(2, 6)  # Testing with 2 to 5 states
log_likelihoods = {'Office': [], 'Delhi': []}

for num_states in state_range:
    # Fit HMM to office data
    _, log_likelihood_office = fit_hmm(office_temp, num_states)
    log_likelihoods['Office'].append(log_likelihood_office)

    # Fit HMM to Delhi data
    _, log_likelihood_delhi = fit_hmm(delhi_temp, num_states)
    log_likelihoods['Delhi'].append(log_likelihood_delhi)

# Plot the log likelihoods for comparison
states_labels = [f'{num_states} states' for num_states in state_range]
x = np.arange(len(states_labels))  # the label locations

fig, ax = plt.subplots()
bar_width = 0.35
bar1 = ax.bar(x - bar_width / 2, log_likelihoods['Office'], bar_width, label='Office')
bar2 = ax.bar(x + bar_width / 2, log_likelihoods['Delhi'], bar_width, label='Delhi')

# Add labels, title, and legend
ax.set_xlabel('Number of States')
ax.set_ylabel('Log Likelihood')
ax.set_title('HMM Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(states_labels)
ax.legend()


# Attach log likelihood values above the bars
def attach_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


attach_values(bar1)
attach_values(bar2)

plt.show()
