import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

n = int(input())
i = 1
s = n - i
r = 0
beta = .3
gamma = .1
v = .175 / 365
imm = -(np.log(.5) / 270)
days = int(input())

susceptible = []
infected = []
recovered = []

for _ in range(days):
  s -= ((beta * i * s) / n) + (v * s) - (imm * r)
  i += ((beta * i * s) / n) - (gamma * i)
  r += (gamma * i) + (v * s) - (imm * r)
  susceptible.append(s)
  infected.append(i)
  recovered.append(r)

sns.lineplot(x = np.arange(days), y = susceptible)
sns.lineplot(x = np.arange(days), y = infected)
sns.lineplot(x = np.arange(days), y = recovered)
plt.title('SIR Model')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend(['Susceptible', 'Infected', 'Recovered'])
plt.show()