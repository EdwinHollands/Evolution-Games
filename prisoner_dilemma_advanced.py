import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# We start with a payoff matrix
# Payoff matrix for prisoner's dilemma
payoff_matrix = np.array([[0, 2], [-1, 1]])
#Now we handle an interaction between two tactics
def payoff(tact1, tact2, payoff_matrix):
    """Returns the payoff of an interaction between two tactics."""
    return payoff_matrix[tact1, tact2]
# Now let's have repeated interactions with adaptable attitudes.
# An individual is an index and list of attitudes, with higher chance of cooperation with preferred players.
# The attitude is the probability of cooperation with another.
#lets start with neutral population with 0.5 preference for everyone
def neutral_attitudes(size):
    """Returns a population of individuals with neutral attitudes."""
    neutral = [0.5 for i in range(size)]
    return np.array([neutral[:i]+[0]+neutral[i+1:] for i in range(size)])
#Let's find how two individuals feel about each other
def relationship(individual1, individual2, attitudes):
    """Returns relative probabilities of cooperation between two individuals."""
    i1_pref_i2 = attitudes[individual1][individual2]
    i2_pref_i1 = attitudes[individual2][individual1]
    return i1_pref_i2, i2_pref_i1
#Let's initiate a neutral fitness
def neutral_fitness(attitudes):
    """Returns fitness 1 for all of a population of individuals."""
    return np.ones(len(attitudes))
#Now we need to handle tactic selection for a pair of individuals
#tactic 1 is cooperation, 0 is defection
def tactic(individual1, individual2, attitudes):
    """Returns the tactic of an individual in a pair."""
    i1_pref_i2 = relationship(individual1, individual2, attitudes)[0]
    return 1 if np.random.rand() < i1_pref_i2 else 0
#We need to model an interaction between two individuals
def adjust_attitudes(individual1, individual2, tact1, tact2, attitudes):
    """Returns updated attitudes based on an interaction."""
    i1_pref_i2, i2_pref_i1 = relationship(individual1, individual2, attitudes)
    i1_pref_i2 = (i1_pref_i2 + tact2)/2
    i2_pref_i1 = (i2_pref_i1 + tact1)/2
    attitudes[individual1][individual2] = i1_pref_i2
    attitudes[individual2][individual1] = i2_pref_i1
    return attitudes
def adjust_fitness(individual1, individual2, tact1, tact2, fitness):
    """Returns updated fitness based on an interaction."""
    payoff1 = payoff(tact1, tact2, payoff_matrix)
    payoff2 = payoff(tact2, tact1, payoff_matrix)
    fitness[individual1] += payoff1
    fitness[individual2] += payoff2
    return fitness
def interact(individual1, individual2, attitudes, fitness):
    """Runs an interaction between two individuals in the population."""
    tact1 = tactic(individual1, individual2, attitudes)
    tact2 = tactic(individual2, individual1, attitudes)
    attitudes = adjust_attitudes(individual1, individual2, tact1, tact2, attitudes)
    fitness = adjust_fitness(individual1, individual2, tact1, tact2, fitness)
    return attitudes, fitness
#Let's now run several random interactions
def interactions(attitudes, fitness, n):
    """Runs n interactions between random pairs of individuals in the population."""
    for _ in range(n):
        individual1 = np.random.randint(0, len(attitudes))
        individual2 = np.random.randint(0, len(attitudes)-1)
        individual2 = individual2 + 1 if individual2 >= individual1 else individual2
        attitudes, fitness = interact(individual1, individual2, attitudes, fitness)
    return attitudes, fitness
#Let's plot a graph of the friendship network
def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    pos = nx.spring_layout(gr)  # Generate positions for the nodes
    nx.draw(gr, pos, node_size=500, labels=mylabels, with_labels=True)
    nx.draw_networkx_labels(gr, pos, labels=mylabels)
    plt.show()
# Let's plot fitness as a function of friendliness
# Fit a line to the data
def fit_line(data):
    x = data[:, 0]
    y = data[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c
# Plot the data and the line
def plot_line(data):
    m, c = fit_line(data)
    plt.scatter(data[:, 0], data[:, 1])
    plt.plot(data[:, 0], m * data[:, 0] + c, color="red")
    plt.xlabel("Friendliness")
    plt.ylabel("Fitness")
    plt.title("How do friendship and fitness relate?")
    plt.show()
#define the friendship network
def friendships(attitudes):
    return np.int64(np.round(attitudes,0))
#define the general disposition of the individuals
def friendliness(attitudes):
    return np.round([sum(friendships(attitudes)[i])/len(attitudes) for i in range(len(attitudes))],2)

def test1(pop, iterations):
    print('new test')
    attitudes = neutral_attitudes(pop)
    fitness = neutral_fitness(attitudes)
    outcome = interactions(attitudes, fitness,iterations)
    friendship = friendships(outcome[0])
    labels = {i: f"{i}" for i in range(len(friendship))}
    relative_fitness = np.round(outcome[1]/sum(outcome[1]),2)
    data = np.array([friendliness(outcome[0]), relative_fitness]).T
    show_graph_with_labels(friendship, labels)
    plot_line(data)


#Now we need to handle reproduction based on fitness
def reproduce(attitudes, fitness):
    """Returns a new population based on fitness."""
    pop_size = len(attitudes)
    new_pop = []
    for i in range(pop_size):
        parent = np.random.choice(range(pop_size), p=fitness/sum(fitness))
        disposition = friendliness(attitudes)[parent]
        child = [disposition for _ in range(pop_size)]
        child[i] = 0
        new_pop.append(child)
    return np.array(new_pop)

#test of generations
def generate(pop, iterations, generations):
    attitudes = neutral_attitudes(pop)
    fitness = neutral_fitness(attitudes)
    if generations >= 2:
        for _ in range(generations-1):
            outcome = interactions(attitudes, fitness, iterations)
            attitudes = outcome[0]
            fitness = outcome[1]
            print(fitness/sum(fitness))
            attitudes = reproduce(attitudes, fitness)
    outcome = interactions(attitudes, fitness, iterations)
    return outcome

def plot_fitness_friendliness(pop, iterations, generations):
    outcome = generate(pop, iterations, generations)
    attitudes = outcome[0]
    fitness = outcome[1]
    friendship = friendships(attitudes)
    labels = {i: f"{i}" for i in range(len(friendship))}
    relative_fitness = np.round(fitness/sum(fitness),7)
    data = np.array([friendliness(attitudes), relative_fitness]).T
    #show_graph_with_labels(friendship, labels)
    plot_line(data)


plot_fitness_friendliness(10, 10000,2)
