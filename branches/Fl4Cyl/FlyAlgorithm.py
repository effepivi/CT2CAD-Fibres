import numpy
import math

from Individual import *


class FlyAlgorithm:
    # Class attributes (same as static attributes in C++)
    
    
    
    # Default constructor
    def __init__(self):
        # Instance attributes
        self.__tournament_size = 2;
        self.__population = []; # Individuals
        self.__mutation_probability = 0.0;
        self.__crossover_probability = 0.7;
        self.__mutation_rate = 0.05;
        self.__new_blood_probability = 0.20;
        self.__elitism_probability = 0.10;
        self.__global_fitness = 0;
        self.__best_global_fitness = 0;
        self.__p_local_fitness  = [];
        self.__sort_index = [];
        self.__fitness_up_to_date = True;
        self.__global_fitness_function = 0;
        self.__use_marginal_fitness = False;
        self.__use_threshold_selection = False;
        self.__use_sharing = False;
         
        Individual.m_fly_algorithm = self;
        
    def setUseSharing(self, aFlag = True):
        self.__use_sharing = aFlag;

    def getUseSharing(self):
        return (self.__use_sharing);

    def setUseTournamentSelection(self, aFlag = True):
        self.__use_threshold_selection = not aFlag;

    def getUseTournamentSelection(self):
        return (not self.__use_threshold_selection);

    def setUseThresholdSelection(self, aFlag = True):
        self.__use_threshold_selection = aFlag;

    def getUseThresholdSelection(self):
        return (self.__use_threshold_selection);

    def setMutationProbability(self, aProbability):
        self.__mutation_probability = aProbability;
    
    def setCrossoverProbability(self, aProbability):
        self.__crossover_probability = aProbability;
    
    def setNewBloodProbability(self, aProbability):
        self.__new_blood_probability = aProbability;
    
    def setElitismProbability(self, aProbability):
        self.__elitism_probability = aProbability;
    
    def setGlobalFitnessFunction(self, f):
        self.__global_fitness_function = f;
        
    def setLocalFitnessFunction(self, f):
        Individual.setFitnessFunction(f);
        
    def setNumberOfIndividuals(self, n, k):
        if len(self.__population):
            raise Exception("The population size has already been set");
        else:
            Individual.setNumberOfGenes(k);
            self.__population    = [ Individual(i) for i in range(n)]
            self.__p_local_fitness = [ 0.0 for i in range(n)]
            self.__fitness_up_to_date = False;

    def getLocalFitnessSet(self):
        return (self.__p_local_fitness);
         
    def restart(self):
        # Update fitnesses if needed
        self.computePopulationFitnesses();
        
        # Get the ID of the best individual
        best_individual_id = self.getBestIndividualId();
        
        # Replace all the individuals, but the best one, by new random individuals
        for i in range(self.getPopulationSize()):
            if i != best_individual_id:
                self.__population[i] = Individual(i);

        # The local fitnesses are not up-to-date
        self.__fitness_up_to_date = False;

    def doublePopulationSize(self):
        # Get the initial population size
        population_size = self.getPopulationSize();
        
        # For each individual in the current population, add a new random one
        for i in range(population_size):
            # Add the new individual to the population
            self.__population.append(Individual(i + population_size));
            self.__p_local_fitness.append(0);
            self.__sort_index = numpy.append(self.__sort_index, 0);

        # The local fitnesses are not up-to-date
        self.__fitness_up_to_date = False;

    def mitosis(self):
        # Get the initial population size
        population_size = self.getPopulationSize();
        
        # For each individual in the current population, add a new one by mutation
        for parent_id in range(population_size):
            # Copy the parent into the child
            child = Individual(parent_id + population_size, self.getIndividual(parent_id));

            # Mutate the child (with a high mutation rate
            child.mutate(self.__mutation_rate * 2.0);

            # Add the child to the population
            self.__population.append(child)
            self.__p_local_fitness.append(0);
            self.__sort_index = numpy.append(self.__sort_index, 0);

        # The local fitnesses are not up-to-date
        self.__fitness_up_to_date = False;
        self.computePopulationFitnesses();

    def getLocalFitnessSet(self):
        return self.__p_local_fitness;

    def getSortedIndexSet(self):
        return self.__sort_index;

    def getNumberOfIndividuals(self):
        return (len(self.__population));
     
    def getPopulationSize(self):
        return (self.getNumberOfIndividuals());
    
    def evolveGeneration(self, n = 1):
    
        # Number of individuals to be created by elitism
        elitism_total = 0;
        if abs(self.__elitism_probability)>0.0001:
            
            # There is more than 1 individual
            if self.getPopulationSize() > 1:
                elitism_total = max(1, math.floor(self.__elitism_probability * self.getPopulationSize()));        
        
        # Create n new generations of offspring
        for i in range(n):
        
            # Update fitnesses if needed
            self.computePopulationFitnesses();
                       
            # Create an empty population of offspring
            p_offspring = [];
            
            # Populate the population of offspring
            for j in range(self.getPopulationSize()):
                
                has_used_elitism = False;

                # Apply elitism
                if self.__use_marginal_fitness:
                    if self.__p_local_fitness[j] > 0.0:
                        has_used_elitism = True;
                        p_offspring.append(Individual(j, self.getIndividual(j)));
                        #print(j, "i s good.");


                elif j < elitism_total:
                    has_used_elitism = True
                    p_offspring.append(Individual(j, self.getIndividual(self.__sort_index[self.getPopulationSize() - j - 1])));

                # Select a genetic operator
                if not has_used_elitism:
                    genetic_operator = random.uniform(0, self.__crossover_probability + self.__mutation_probability + self.__new_blood_probability);
                
                    # Apply crossover
                    if genetic_operator <= self.__crossover_probability:
                        parent_id1 = self.selectIndividual(True);
                        parent_id2 = self.selectIndividual(True);
                        child = Individual(j);
                        child.crossOver(self.getIndividual(parent_id1), self.getIndividual(parent_id2));
                        child.mutate(self.__mutation_rate);
                        p_offspring.append(child);
                    # Apply mutation
                    elif genetic_operator <= self.__crossover_probability + self.__mutation_probability:
                        parent_id = self.selectIndividual(True);
                        child = Individual(j, self.getIndividual(parent_id));
                        child.mutate(self.__mutation_rate);
                        p_offspring.append(child);
                    # New blood
                    else: #if genetic_operator <= self.__crossover_probability + self.__mutation_probability + self.__new_blood_probability:
                        p_offspring.append(Individual(j));
        
            # Replace the parents by the offspring
            self.__population = copy.deepcopy(p_offspring);
            self.__fitness_up_to_date = False;
            self.computePopulationFitnesses();
        
    def evolveSteadyState(self, n = 1):
    
        # Create n new generations of offspring
        for i in range(n):
        
            # Populate the population of offspring
            for j in range(self.getPopulationSize()):
                
                genetic_operator = random.uniform(0, self.__crossover_probability + self.__mutation_probability + self.__new_blood_probability);
           
                bad_individual_id = self.selectIndividual(False);
                #print("Bad fly: ", bad_individual_id)
                #self.__population[bad_individual_id].print();
                # Apply crossover
                if genetic_operator <= self.__crossover_probability:
                    parent_id1 = self.selectIndividual(True);
                    parent_id2 = self.selectIndividual(True);
                    self.__population[bad_individual_id].crossOver(self.getIndividual(parent_id1), self.getIndividual(parent_id2));
                    self.__population[bad_individual_id].mutate(self.__mutation_rate);
                # Apply mutation
                elif genetic_operator <= self.__crossover_probability + self.__mutation_probability:
                    parent_id = self.selectIndividual(True);
                    self.__population[bad_individual_id] = Individual(bad_individual_id, self.getIndividual(parent_id));
                    self.__population[bad_individual_id].mutate(self.__mutation_rate);
                # New blood
                else: #if genetic_operator <= self.__crossover_probability + self.__mutation_probability + self.__new_blood_probability:
                    self.__population[bad_individual_id] = Individual(bad_individual_id);
        
                #self.__population[bad_individual_id].print();
                #print()
                
                self.__fitness_up_to_date = False;
                self.computeGlobalFitness();

    def kill(self, i):
        self.__population[i] = Individual(i);
        self.__fitness_up_to_date = False;
        #self.computeGlobalFitness();
        



    # Accessor on the population
    def getPopulation(self):
        return (self.__population);
    
    # Compute the global fitness
    def computeGlobalFitness(self):
        self.__global_fitness = self.__global_fitness_function(self.__population);
        
        if self.__best_global_fitness < self.__global_fitness:
            self.__best_global_fitness = self.__global_fitness;
            #for ind in self.__population:
            #    ind.print()
        
        return self.__global_fitness;
        
    # Compute the local fitness
    def computeLocalFitness(self, i):
        self.__p_local_fitness[i] = self.__population[i].computeFitness();
        return (self.__p_local_fitness[i]);

    # Get the globalfitness (no new computation)
    def getGlobalFitness(self):
        return self.__global_fitness;
        
    # Get the local fitness (no new computation)
    def getLocalFitness(self, i):
        if self.__fitness_up_to_date == False:
            self.__p_local_fitness[i] = self.computeLocalFitness(i);
        return self.__p_local_fitness[i];
    
    def computePopulationFitnesses(self):
        if self.__fitness_up_to_date == False:
        
            if self.__global_fitness_function:
                self.computeGlobalFitness();
                #print ("Global fitness:\t", self.__global_fitness * 100, "%");
        
            for i in range(self.getPopulationSize()):
                self.computeLocalFitness(i);

            self.__sort_index = numpy.argsort(self.__p_local_fitness)

            self.__fitness_up_to_date = True;
        
            
    # Accessor on an individual
    def getIndividual(self, i):
        return self.__population[i]
    
    def thresholdSelection(self, aFlag):
        counter = 0;
        not_found = True;
        ind_id = 0;

        while (not_found and counter <= self.getPopulationSize() * 0.5):
            
            ind_id = random.randint(0, self.getPopulationSize() - 1);

            if self.__use_sharing:
                fitness = self.fitnessSharing(ind_id, 1.0 / 10.0);
            else:
                fitness = self.getLocalFitness(ind_id);

            if (aFlag and fitness > 0.0):
                not_found = False;
            elif (not aFlag and fitness < 0.0):
                not_found = False;

            counter += 1

        if not_found:
            ind_id = self.tournamentSelection(aFlag);

        return ind_id;


    # Tournament selection.
    # If aFlag is True return the id of the highest fitness amongst 
    # self.__tournament_size randomly chosen individuals
    # If aFlag is False return the id of the lowest fitness amongst 
    # self.__tournament_size randomly chosen individuals
    def tournamentSelection(self, aFlag):
        best_id = random.randint(0, self.getPopulationSize() - 1);

        if self.__use_sharing:
            best_fitness = self.fitnessSharing(best_id, 1.0 / 10.0);
        else:
            best_fitness = self.getLocalFitness(best_id);
        
        for i in range(self.__tournament_size - 1):
            new_id = random.randint(0, self.getPopulationSize() - 1);

            if self.__use_sharing:
                new_fitness = self.fitnessSharing(new_id, 1.0 / 10.0);
            else:
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

    def selectIndividual(self, aFlag):
        if (self.__use_threshold_selection):
            return (self.thresholdSelection(aFlag));
        else:
            return (self.tournamentSelection(aFlag));
        
    def getBestIndividualId(self):
        self.computePopulationFitnesses();
        
        for i in range(self.getPopulationSize(), 0, -1):
            if self.__population[self.__sort_index[i - 1]].isActive():
                return (self.__sort_index[i - 1]);

        return -1;
        #return self.__sort_index[self.getPopulationSize() - 1];
        
    def getWorseIndividualId(self):
        self.computePopulationFitnesses();

        for i in range(self.getPopulationSize()):
            if self.__population[self.__sort_index[i]].isActive():
                return (self.__sort_index[i]);

        return -1;
        #return self.__sort_index[0];
                
    def getBestIndividual(self):
        self.computePopulationFitnesses();
        return self.getIndividual(self.getBestIndividualId());
        
    def getWorseIndividual(self):
        self.computePopulationFitnesses();
        return self.getIndividual(self.getWorseIndividualId());

    def setUseMarginalFitness(self, aFlag):
        self.__use_marginal_fitness = aFlag;

    def getUseMarginalFitness(self):
        return (self.__use_marginal_fitness);

    def sharing(self, i1, i2, aRadius):
        distance = 0.0;
        for g1, g2 in zip(self.getIndividual(i1).getGeneSet(), self.getIndividual(i2).getGeneSet()):
            distance += (g1 - g2) * (g1 - g2);

        distance = math.sqrt(distance);

        if (distance < 2 * aRadius):
            sharing = 1.0 - (distance / (2.0 * aRadius));
        else:
            sharing = 0;

        return sharing;

    def fitnessSharing(self, i, aRadius):
        local_fitness = self.getLocalFitness(i);
        
        sum_sharing = 0;
        
        for j in range(self.getPopulationSize()):
            if i != j:
                sum_sharing += self.sharing(i, j, aRadius);

        return (local_fitness / sum_sharing);

    def isIndividualActive(self, i):
        return (self.__population[i].isActive());

    def tempActivateIndividual(self, i):
        self.__population[i].activate();

    def tempDeactivateIndividual(self, i):
        self.__population[i].deactivate();

    def activateIndividual(self, i):
        self.__population[i].activate();
        self.__fitness_up_to_date = False;

    def deactivateIndividual(self, i):
        self.__population[i].deactivate();
        self.__fitness_up_to_date = False;

    def getPopulation(self):
        return (self.__population);

    def setPopulation(self, aPopulation):
        self.__population = copy.deepcopy(aPopulation);
        self.__fitness_up_to_date = False;

