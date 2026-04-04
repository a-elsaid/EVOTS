from .individual import Individual

class Population:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.individuals = []  # list of Individual

    def add(self, indiv: Individual):
        if len(self.individuals) < self.max_size:
            self.individuals.append(indiv)
        else:
            raise RuntimeError("Population is full")

    def remove(self, indiv: Individual):
        self.individuals = [x for x in self.individuals if x.id != indiv.id]

    def size(self):
        return len(self.individuals)

    def get_all(self):
        return list(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def best_fitness(self):
        """Return the lowest fitness among individuals with a defined fitness."""
        inds = self.get_all()
        fits = [ind.fitness for ind in inds if ind.fitness is not None]
        if not fits:
            return float("inf")
        return min(fits)
    
    def best_individual(self):
        """Return the best individual in the population."""
        inds = self.get_all()
        best_ind = None
        best_fit = float("inf")
        for ind in inds:
            if ind.fitness is not None and ind.fitness < best_fit:
                best_ind = ind
                best_fit = ind.fitness
        return best_ind