import numpy as np
import matplotlib.pyplot as plt
# We start with a payoff matrix
# Payoff matrix for prisoner's dilemma
payoff_matrix = np.array([[[2, 2], [0, 3]], [[3, 0], [1, 1]]])
print(f"Payoff matrix for prisoner's dilemma:\n"
      "So if both Cooperate, they get 2 each.\n"
      "If both Defect, they get 1 each.\n"
      "If one Cooperates and the other Defects, the one who Defects gets 3 and the one who Cooperates gets 0.")