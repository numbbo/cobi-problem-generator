from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population


class GeneticAlgorithmUnsortedPop(GeneticAlgorithm):
    """ Adaptation of the simple Genetic Algorithm from Pymoo that keeps track of the unsorted population. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unsorted_pop = None

    def _initialize_advance(self, infills=None, **kwargs):
        self.unsorted_pop = self.pop.copy()
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills), algorithm=self, **kwargs)

    def _advance(self, infills=None, **kwargs):
        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.unsorted_pop = pop.copy()

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)
