import numpy
import math

from Individual import *


class FlyAlgorithm:
    # Class attributes (same as static attributes in C++)
    
    
    
    # Default constructor
    def __init__(self):
        # Instance attributes
        self.m_tournament_size = 2;
        self.m_p_population = []; # Individuals
        self.m_mutation_probability = 0.0;
        self.m_crossover_probability = 0.7;
        self.m_mutation_rate = 0.05;
        self.m_new_blood_probability = 0.20;
        self.m_elitism_probability = 0.10;
        self.m_p_global_fitness = 0;
        self.m_best_global_fitness = 0;
        self.m_p_local_fitness  = [];
        self.m_sort_index = [];
        self.m_fitness_up_to_date = True;
        self.m_global_fitness_function = 0;
        
        Individual.m_fly_algorithm = self;
        
    
    def setGlobalFitnessFunction(self, f):
        self.m_global_fitness_function = f;
        
    def setLocalFitnessFunction(self, f):
        Individual.setFitnessFunction(f);
        
    def setNumberOfIndividuals(self, n, k):
        if len(self.m_p_population):
            raise Exception("The population size has already been set");
        else:
            Individual.setNumberOfGenes(k);
            self.m_p_population    = [ Individual(i) for i in range(n)]
            self.m_p_local_fitness = [ 0.0 for i in range(n)]
            self.m_fitness_up_to_date = False;
 
    def restart(self):
        # Update fitnesses if needed
        self.computePopulationFitnesses();
        
        # Get the ID of the best individual
        best_individual_id = self.getBestIndividualId():
        
        # Replace all the individuals, but the best one, by new random individuals
        for i in range(self.getPopulationSize()):
            if i != best_individual_id:
                self.m_p_population[i] = Individual(i);

        # The local fitnesses are not up-to-date
        self.m_fitness_up_to_date = False;

    def getNumberOfIndividuals(self):
        return (len(self.m_p_population));
     
    def getPopulationSize(self):
        return (self.getNumberOfIndividuals());
    
    def evolve(self, n = 1):
    
        # Number of individuals to be created by elitism
        elitism_total = 0;
        if abs(self.m_elitism_probability)>0.0001:
            
            # There is more than 1 individual
            if self.getPopulationSize() > 1:
                elitism_total = max(1, math.floor(self.m_elitism_probability * self.getPopulationSize()));
        
        
        # Create n new generations of offspring
        for i in range(n):
        
            # Update fitnesses if needed
            self.computePopulationFitnesses();
                       
            # Create an empty population of offspring
            p_offspring = [];
            
            # Populate the population of offspring
            for j in range(self.getPopulationSize()):
                
                # Apply elitism
                if j < elitism_total:
                    p_offspring.append(Individual(j, self.getIndividual(self.m_sort_index[self.getPopulationSize() - j - 1])));
                    
                # Select a genetic operator
                else:
                    genetic_operator = random.uniform(0, self.m_crossover_probability + self.m_mutation_probability + self.m_new_blood_probability);
                
                    # Apply crossover
                    if genetic_operator <= self.m_crossover_probability:
                        parent_id1 = self.tournament(True);
                        parent_id2 = self.tournament(True);
                        child = Individual(j);
                        child.crossOver(self.getIndividual(parent_id1), self.getIndividual(parent_id2));
                        child.mutate(self.m_mutation_rate);
                        p_offspring.append(child);
                    # Apply mutation
                    elif genetic_operator <= self.m_crossover_probability + self.m_mutation_probability:
                        parent_id = self.tournament(True);
                        child = Individual(j, self.getIndividual(parent_id));
                        child.mutate(self.m_mutation_rate);
                        p_offspring.append(child);
                    # New blood
                    else: #if genetic_operator <= self.m_crossover_probability + self.m_mutation_probability + self.m_new_blood_probability:
                        p_offspring.append(Individual(j));
        
            # Replace the parents by the offspring
            self.m_p_population = copy.deepcopy(p_offspring);
            self.m_fitness_up_to_date = False;
        
    # Accessor on the population
    def getPopulation(self):
        return (self.m_p_population);
    
    # Compute the global fitness
    def computeGlobalFitness(self):
        self.m_global_fitness = self.m_global_fitness_function(self.m_p_population);
        
        if self.m_best_global_fitness < self.m_global_fitness:
            self.m_best_global_fitness = self.m_global_fitness;
            #for ind in self.m_p_population:
            #    ind.print()
        
        return self.m_global_fitness;
        
    # Compute the local fitness
    def computeLocalFitness(self, i):
        self.m_p_local_fitness[i] = self.m_p_population[i].computeFitness();
        return (self.m_p_local_fitness[i]);

    # Get the globalfitness (no new computation)
    def getGlobalFitness(self):
        return self.m_global_fitness;
        
    # Get the local fitness (no new computation)
    def getLocalFitness(self, i):
        return self.m_p_local_fitness[i];
    
    def computePopulationFitnesses(self):
        if self.m_fitness_up_to_date == False:
        
            if self.m_global_fitness_function:
                self.computeGlobalFitness();
                print ("Global fitness:\t", self.m_global_fitness * 100, "%");
        
            for i in range(self.getPopulationSize()):
                self.computeLocalFitness(i);

            self.m_sort_index = numpy.argsort(self.m_p_local_fitness)

            self.m_fitness_up_to_date = True;
        
            
    # Accessor on an individual
    def getIndividual(self, i):
        return self.m_p_population[i]
    
    # Tournament selection.
    # If aFlag is True return the id of the highest fitness amongst 
    # self.m_tournament_size randomly chosen individuals
    # If aFlag is False return the id of the lowest fitness amongst 
    # self.m_tournament_size randomly chosen individuals
    def tournament(self, aFlag):
        best_id = random.randint(0, self.getPopulationSize() - 1);
        best_fitness = self.getLocalFitness(best_id);
        
        for i in range(self.m_tournament_size - 1):
            new_id = random.randint(0, self.getPopulationSize() - 1);
            new_fitness = self.getLocalFitness(new_id);
            
            if aFlag:
                if best_fitness < new_fitness:
                    best_id = new_id;
                    best_fitness = new_fitness;
            else:
                if best_fitness > new_fitness:
                    best_id = new_id;
                    best_fitness = new_fitness;
        
        return best_id;

        
    def getBestIndividualId(self):
        self.computePopulationFitnesses();
        return self.m_sort_index[self.getPopulationSize() - 1];
        
    def getWorseIndividualId(self):
        self.computePopulationFitnesses();
        return self.m_sort_index[0];
                
    def getBestIndividual(self):
        self.computePopulationFitnesses();
        return self.getIndividual(self.getBestIndividualId());
        
    def getWorseIndividual(self):
        self.computePopulationFitnesses();
        return self.getIndividual(self.getWorseIndividualId());
