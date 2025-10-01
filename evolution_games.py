import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy

class Creature:
    """A creature in the population."""
    def __init__(self,name):
        self.name = name
        self.fitness = 1
        self.tactic = None
        self.strategy = None
    def mutate(self):
        self.fitness += np.random.normal(0,0.1)
    def select_tactic(self,opponent):
        self.tactic = self.strategy(self,opponent)

class Population:
    """A population of creatures."""
    def __init__(self, size):
        self.size = size
        self.creatures = [Creature(i) for i in range(size)]
    def strategies(self):
        return [creature.strategy.__name__ for creature in self.creatures]
    def fitnesses(self):
        return np.array([creature.fitness for creature in self.creatures])
    def total_fitness(self):
        return np.sum(self.fitnesses())
    def random_creature(self):
        return np.random.choice(self.creatures)
    def random_creature_except(self, creature):
        creatures = self.creatures.copy()
        creatures.remove(creature)
        return np.random.choice(creatures)
    def n_distict_creatures(self,n):
        return np.random.choice(self.creatures,n,replace=False)
    def random_creatures_except(self, creature, n):
        return [self.random_creature_except(creature) for _ in range(n)]
    def choose_fit_creature(self):
        return np.random.choice(self.creatures, p=self.fitnesses()/self.total_fitness())
    def reproduce(self):
        parent = self.choose_fit_creature()
        child = copy.copy(parent)
        child.mutate()
        self.creatures.append(child)
    def replacement(self):
        if self.size == 0:
            return
        creature = self.random_creature()
        self.creatures.remove(creature)
        self.reproduce()
    def replacements(self, n):
        for _ in range(n):
            self.replacement()
    def time_until_fixation(self):
        pop = copy.deepcopy(self)
        time = 0
        while len(set(pop.tactics())) > 1:
            pop.replacement()
            time += 1
        return time
    def time_until_fixation_average(self, n):
        times = [self.time_until_fixation() for _ in range(n)]
        return np.mean(times)
    def play_game(self, game, creature1, creature2):
        game(creature1, creature2)
    def round_robin(self, game):
        for creature1 in self.creatures:
            for creature2 in self.creatures:
                if creature1 != creature2:
                    self.play_game(game, creature1, creature2)
    def round_robins(self, game, n):
        for _ in range(n):
            self.round_robin(game)
    def evolve(self, game, robins, generations):
        for _ in range(generations):
            self.round_robins(game, robins)
            self.replacement()
            for creature in self.creatures:
                creature.fitness = 1

#Define strategies as functions that return tactics
def Defector(creature1,creature2):
    return 'Defect'
def Cooperator(creature1,creature2):
    return 'Cooperate'
def Hawk(creature1,creature2):
    return 'Hawk'
def Dove(creature1,creature2):
    return 'Dove'

#Define games
def prisoner_dilemma(creature1,creature2):
    creature1.select_tactic(creature2)
    creature2.select_tactic(creature1)
    if creature1.tactic == 'Cooperate' and creature2.tactic == 'Cooperate':
        creature1.fitness += 3
        creature2.fitness += 3
    elif creature1.tactic == 'Cooperate' and creature2.tactic == 'Defect':
        creature1.fitness += 0
        creature2.fitness += 5
    elif creature1.tactic == 'Defect' and creature2.tactic == 'Cooperate':
        creature1.fitness += 5
        creature2.fitness += 0
    elif creature1.tactic == 'Defect' and creature2.tactic == 'Defect':
        creature1.fitness += 1
        creature2.fitness += 1
def hawk_dove(creature1,creature2):
    creature1.select_tactic(creature2)
    creature2.select_tactic(creature1)
    if creature1.tactic == 'Dove' and creature2.tactic == 'Dove':
        creature1.fitness += 3
        creature2.fitness += 3
    elif creature1.tactic == 'Dove' and creature2.tactic == 'Hawk':
        creature1.fitness += 1
        creature2.fitness += 5
    elif creature1.tactic == 'Hawk' and creature2.tactic == 'Dove':
        creature1.fitness += 5
        creature2.fitness += 1
    elif creature1.tactic == 'Hawk' and creature2.tactic == 'Hawk':
        creature1.fitness += 0
        creature2.fitness += 0    
def PD_plot_defector_takeover(iterations,starting_defectors,pop_size,max_generations):
    for _ in range(iterations):
        x = Population(pop_size)
        n=0
        for creature in x.creatures:
            if n<starting_defectors:
                creature.strategy = Defector
                n+=1
            else:
                creature.strategy = Cooperator
        Defectors = []
        for _ in range(max_generations):
            if len(set(x.strategies())) == 1:
                break
            D = len([creature for creature in x.creatures if creature.tactic == 'Defect'])
            Defectors.append(D)
            x.evolve(prisoner_dilemma,1,1)
        t = np.arange(0,len(Defectors),1)
        plt.plot(t,Defectors)
    plt.show()
def HD_plot_defector_takeover(iterations,starting_hawks,pop_size,max_generations):
    for _ in range(iterations):
        x = Population(pop_size)
        n=0
        for creature in x.creatures:
            if n<starting_hawks:
                creature.strategy = Hawk
                n+=1
            else:
                creature.strategy = Dove
        Hawks = []
        for _ in range(max_generations):
            if len(set(x.strategies())) == 1:
                break
            D = len([creature for creature in x.creatures if creature.tactic == 'Hawk'])
            Hawks.append(D)
            x.evolve(prisoner_dilemma,1,1)
        t = np.arange(0,len(Hawks),1)
        plt.plot(t,Hawks)
    plt.show()






PD_plot_defector_takeover(10,5,10,1000)