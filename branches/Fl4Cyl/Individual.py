import random
import copy
from FlyAlgorithm import *

class Individual:
    m_number_of_genes = 0;
    m_fitness_function = 0;
    m_fly_algorithm = 0;

    # Default and copy constructor
    def __init__(self, ind_id=-1, other=None):
        self.m_id = ind_id;
        self.m_fitness = 0;
        self.m_is_active = True;
        
        if other is None:
            self.defaultConstructor();
        else:
            self.copyConstructor(other);
    
    def defaultConstructor(self):
        if (Individual.m_number_of_genes):
            self.createRandomIndividual();
    
    def copyConstructor(self, anIndividual):
        self.m_p_gene_set = copy.deepcopy(anIndividual.m_p_gene_set)
        self.m_is_active  = anIndividual.m_is_active;
    
    def createRandomIndividual(self):
        self.m_p_gene_set = [random.uniform(0.5-0.275,0.5+0.275) for _ in range(Individual.getNumberOfGenes())];
    
    def setNumberOfGenes(n):
        Individual.m_number_of_genes = n;

    def setFitnessFunction(f):
        Individual.m_fitness_function = f;

    def getNumberOfGenes(self):
        return (Individual.m_number_of_genes);
        
    def getNumberOfGenes():
        return (Individual.m_number_of_genes);

    def computeFitness(self):
        self.m_fitness = Individual.m_fitness_function(self.m_id, self.m_p_gene_set, Individual.m_fly_algorithm);
        return self.m_fitness;
        
    def print(self):
        print(self.m_p_gene_set, self.m_fitness, self.m_is_active);
    
    def immigration(self):
        self.createRandomIndividual();
        
    def newBlood(self):
        self.createRandomIndividual();
            
    def mutate(self, mu):
        for i in range(len(self.m_p_gene_set)):
            self.m_p_gene_set[i] += random.gauss(0.0, mu);
    
    def crossOver(self, ind1, ind2):
        for i in range(len(self.m_p_gene_set)):
            self.m_p_gene_set[i] = self.m_p_gene_set[i] * ind1.m_p_gene_set[i] + (1.0 - self.m_p_gene_set[i]) * ind2.m_p_gene_set[i]                        
        
    def isBetterThan(self, anIndividual):
        return (self.m_fitness > anIndividual.m_fitness);
        
    def getGene(self, i):
        return (self.m_p_gene_set[i]);
