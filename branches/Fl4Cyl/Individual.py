import random
import copy
from FlyAlgorithm import *

class Individual:
    __number_of_genes = 0;
    __fitness_function = 0;
    __m_p_label_set = list();
    m_fly_algorithm = 0;
    
    __p_metrics_label_set        = list();
    __p_metrics_minimisation_set = list();
    __p_metrics_function_set     = list();

    # Default and copy constructor
    def __init__(self, ind_id=-1, other=None):
        self.__id = ind_id;
        self.__fitness = 0;
        self.__is_active = True;
        self.__p_metrics_set = list();
        
        if other is None:
            self.defaultConstructor();
        else:
            self.copyConstructor(other);
    
    def defaultConstructor(self):
        if (Individual.__number_of_genes):
            self.createRandomIndividual();
    
    def copyConstructor(self, anIndividual):
        self.__p_gene_set = copy.deepcopy(anIndividual.__p_gene_set)
        self.__is_active  = anIndividual.__is_active;
        self.__fitness    = anIndividual.__fitness;
    
    def createRandomIndividual(self):
        self.__p_gene_set = [random.uniform(0.5-0.275,0.5+0.275) for _ in range(Individual.getNumberOfGenes())];
    
    def setNumberOfGenes(n):
        Individual.__number_of_genes = n;

        Individual.__m_p_label_set = list();
        for i in range(n):
            Individual.__m_p_label_set.append("gene(" + str(i) + ")");
    
    def setFitnessFunction(f):
        Individual.__fitness_function = f;
    
    def getID(self):
        return self.__id;
    
    def getNumberOfGenes(self):
        return (Individual.__number_of_genes);
    
    def getNumberOfGenes():
        return (Individual.__number_of_genes);
    
    def computeFitness(self):
        self.__fitness = Individual.__fitness_function(self.__id, self.__p_gene_set, Individual.m_fly_algorithm);
        return self.__fitness;
    
    def print(self):
        print(self.__p_gene_set, self.__fitness, self.__is_active);
    
    def immigration(self):
        self.createRandomIndividual();
    
    def newBlood(self):
        self.createRandomIndividual();
    
    def mutate(self, mu):
        for i in range(len(self.__p_gene_set)):
            self.__p_gene_set[i] += random.gauss(0.0, mu);
    
    def crossOver(self, ind1, ind2):
        for i in range(len(self.__p_gene_set)):
            mu = random.gauss(0.0, 1.0)
            self.__p_gene_set[i] = mu * ind1.__p_gene_set[i] + (1.0 - mu) * ind2.__p_gene_set[i]                        
    
    def isBetterThan(self, anIndividual):
        return (self.__fitness > anIndividual.__fitness);
    
    def getGene(self, i):
        return (self.__p_gene_set[i]);
    
    def setGene(self, i, gene):
        self.__p_gene_set[i] = gene;
    
    def getGeneSet(self):
        return (copy.deepcopy(self.__p_gene_set));
    
    def isActive(self):
        return (self.__is_active);
    
    def activate(self):
        self.__is_active = True;
    
    def deactivate(self):
        self.__is_active = False;
    
    def getGeneLabel(i):
        return Individual.__m_p_label_set[i];
    
    def setGeneLabel(i, aLabel):
        Individual.__m_p_label_set[i] = aLabel;
        
    def addMetrics(aLabel, aMaximisationFlag, aFunction):
        Individual.__p_metrics_label_set.append(aLabel);
        Individual.__p_metrics_minimisation_set.append(not aMaximisationFlag);
        Individual.__p_metrics_function_set.append(aFunction);
        
    def computeMetrics(self):

        self.__p_metrics_set = list();
        for metrics in Individual.__p_metrics_function_set:
            self.__p_metrics_set.append(metrics());

    def getMetrics(self, i):
        return self.__p_metrics_set[i];