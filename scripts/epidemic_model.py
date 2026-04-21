"""
This script simulates the SIR (Susceptible, Infected, Recovered) model of epidemic spread.
It takes the population size and the number of days to simulate as input.
Then calculates the number of susceptible, infected, and recovered individuals over time.
This is based on the infection rate, recovery rate, vaccination rate, and immunity loss rate.
Finally, it plots the results using Seaborn and Matplotlib.
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Population size
n = int(input())

# Infected individuals at the start
i = 1

# Susceptible individuals at the start
s = n - i

# Recovered individuals at the start
r = 0

# Infection rate
beta = .3

# Recovery rate
gamma = .1

# Vaccination rate
v = .175 / 365

# Immunity loss rate
imm = -(np.log(.5) / 270)

# Number of days to simulate
days = int(input())

# Initialize lists to store the number of susceptible, infected, and recovered individuals over time
susceptible = []
infected = []
recovered = []

# Simulate the SIR model over the specified number of days
for _ in range(days):
  # Update the number of susceptible, infected, and recovered individuals based on the SIR model equations
  s -= ((beta * i * s) / n) + (v * s) - (imm * r)
  i += ((beta * i * s) / n) - (gamma * i)
  r += (gamma * i) + (v * s) - (imm * r)

  # Append the current number of susceptible, infected, and recovered individuals to the respective lists
  susceptible.append(s)
  infected.append(i)
  recovered.append(r)

# Plot the results
sns.lineplot(x = np.arange(days), y = susceptible)
sns.lineplot(x = np.arange(days), y = infected)
sns.lineplot(x = np.arange(days), y = recovered)
plt.title('SIR Model')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend(['Susceptible', 'Infected', 'Recovered'])
plt.show()