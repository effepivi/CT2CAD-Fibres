#!/usr/bin/env python3
import random
import math
import sys
import os

#from PIL import Image

import numpy as np

# Import the X-ray simulation library
import gvxrPython3 as gvxr

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.INFO)


from skimage.transform import radon, iradon, iradon_sart

from lsf import *

from FlyAlgorithm import *

#import multiprocessing
#from multiprocessing import Pool



################################################################################
# Global variables
################################################################################

##### Parameters of the Fly Algorithm #####
INITIAL_NUMBER_OF_INDIVIDUALS = 5;
NUMBER_OF_MITOSIS = 3;
NUMBER_OF_GENES = 2;
NUMBER_OF_GENERATIONS = 10;

MAX_VALUE = 1217.0;


##### Parameters of the X-ray simulation #####
#detector_width_in_pixels  = math.floor(1024);
#detector_height_in_pixels = math.floor( 615);
detector_width_in_pixels  = 801;
#detector_height_in_pixels = 611;
detector_height_in_pixels = 1;

pixel_size_in_micrometer = 1.9;

distance_source_detector_in_m  = 145.0;
distance_object_detector_in_m =    0.08; # = 80 mm

#number_of_projections = 900;
number_of_projections = 45;
angular_span_in_degrees = 180.0;

#energy_spectrum_in_keV = [(33, 0.97), (66, 0.02), (99, 0.01)];
energy_spectrum_in_keV = [(33, 1.0)];

angular_step = angular_span_in_degrees / number_of_projections;

fiber_radius = 140 / 2; # um
fiber_material = "SiC";
fiber_mu = 2.736; # cm-1
fiber_density = 3.2; # g/cm3

core_radius = 30 / 2; # um
core_material = "W";
core_mu = 341.61; # cm-1
core_density = 19.3 # g/cm3


g_matrix_width = 0;
g_matrix_height = 0;
g_matrix_x = 0;
g_matrix_y = 0;
matrix_material = "Ti90Al6V4";
matrix_mu = 13.1274; # cm-1
matrix_density = 4.42 # g/cm3

g_reference_CT        = np.zeros(1);
g_reference_sinogram  = np.zeros(1);
g_normalised_CT       = np.zeros(1);
g_normalised_sinogram = np.zeros(1);

g_fiber_geometry_set = [];
g_core_geometry_set  = [];

g_best_population = [];

g_fly_algorithm = 0;
g_number_of_evaluations = 0;
g_number_of_generations = 0;
g_number_of_mitosis = 0;
g_best_ncc = 0;

g_best_ncc_sinogram = [];
g_best_ncc_CT_slice = [];

g_first_run = True;


########
def cropCenter(anImage, aNewSizeInX, aNewSizeInY):
    y, x = anImage.shape
    start_x = x // 2 - (aNewSizeInX // 2);
    start_y = y // 2 - (aNewSizeInY // 2);   
    return anImage[start_y : start_y + aNewSizeInY,
            start_x : start_x + aNewSizeInX]


temp_mean = 0.0;
temp_std = 0.0;

def f1(x):
    global temp_mean;
    global temp_std;
    return (x-temp_mean)/temp_std;

def f2(x, y):
    return (x*y);


def productImage(anImage1, anImage2):

    return (np.multiply(anImage1, anImage2));
    
    p = Pool(multiprocessing.cpu_count())
    return (p.map(f2, (anImage1, anImage2)))

def normaliseImage(anImage):
    global temp_mean;
    global temp_std;

    temp_mean = anImage.mean();
    temp_std = anImage.std();

    return (anImage-temp_mean)/temp_std

    p = Pool(multiprocessing.cpu_count())
    return (p.map(f1, anImage))
    

################################################################################
# Fitness functions
################################################################################

########################################################
def globalFitnessFunction(aPopulation, resetAll = True):
########################################################
    
    global g_best_ncc;
    global g_number_of_evaluations;
    global g_best_population;
    global g_reference_sinogram;
    global g_reference_CT;
    global g_normalised_sinogram;
    global g_normalised_CT;
    global g_first_run;
    
    setCylinders(aPopulation, resetAll);    
    sinogram = computeSinogram();
    
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    test_image = iradon(sinogram.T, theta=theta, circle=True);
    

    g_normalised_sinogram = normaliseImage(sinogram);
    g_normalised_CT = normaliseImage(test_image);
    #normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
    #normalised_CT       = (test_image       - test_image.mean())       / test_image.std();

    g_number_of_evaluations += 1;
    
    # Fitness based on NCC
    ncc_sinogram =  productImage(g_reference_sinogram, g_normalised_sinogram).mean();
    ncc_CT       =  productImage(g_reference_CT, g_normalised_CT).mean();

    ncc = (ncc_sinogram + ncc_CT) / 2.0;
    ncc = ncc_CT * ncc_sinogram;
    #ncc = ncc_sinogram
    #ncc = ncc_CT
    
    # Display the 3D scene (no event loop)
    #gvxr.displayScene();

    if g_first_run:
        np.savetxt("sinogram_cylinders_gvxr.txt", g_normalised_sinogram);
        np.savetxt("CT_cylinders_gvxr.txt",       g_normalised_CT);
        g_best_ncc = ncc;
        g_first_run = False;
        g_best_population = copy.deepcopy(aPopulation);
    elif g_best_ncc < ncc:
        np.savetxt("sinogram_cylinders_gvxr.txt", g_normalised_sinogram);
        np.savetxt("CT_cylinders_gvxr.txt",       g_normalised_CT);
        g_best_ncc = ncc;
        g_best_population = copy.deepcopy(aPopulation);

    
    # Fitness based on NCC
    return (ncc);


######################################################
def localFitnessFunction(ind_id, genes, aFlyAlgorithm):
#######################################################

    local_fitness = 0.0;
    global_fitness = aFlyAlgorithm.getGlobalFitness();
    #print("global fitness:\t", global_fitness)

    state = aFlyAlgorithm.isIndividualActive(ind_id);
    
    if state:
        aFlyAlgorithm.tempDeactivateIndividual(ind_id);
        local_fitness = global_fitness - globalFitnessFunction(aFlyAlgorithm.getPopulation(), False);
        aFlyAlgorithm.tempActivateIndividual(ind_id);
    
    #print("local fitness(", ind_id, "):\t", local_fitness)
    return local_fitness;


#######################
def initFlyAlgorithm():
#######################
    
    fly_algorithm = FlyAlgorithm();
    fly_algorithm.setNumberOfIndividuals(INITIAL_NUMBER_OF_INDIVIDUALS, NUMBER_OF_GENES);
    fly_algorithm.setGlobalFitnessFunction(globalFitnessFunction);
    fly_algorithm.setLocalFitnessFunction(localFitnessFunction);
    fly_algorithm.setUseMarginalFitness(True);

    fly_algorithm.setMutationProbability( 0.7);
    fly_algorithm.setCrossoverProbability(0.0);
    fly_algorithm.setNewBloodProbability( 0.0);
    fly_algorithm.setElitismProbability(  0.0);

    fly_algorithm.setUseThresholdSelection(True);
    #fly_algorithm.setUseTournamentSelection(True);
    #fly_algorithm.setUseSharing(True);

    print ("Use marginal fitness:\t", fly_algorithm.getUseMarginalFitness());
    print ("Use threshold selection:\t", fly_algorithm.getUseThresholdSelection());
    print ("Use tournament selection:\t", fly_algorithm.getUseTournamentSelection());
    print ("Use sharing:\t", fly_algorithm.getUseSharing());
    return (fly_algorithm);


########################
def initXRaySimulator():
########################

    # Set up the beam
    print("Set up the beam")
    gvxr.setSourcePosition(distance_source_detector_in_m - distance_object_detector_in_m,  0.0, 0.0, "mm");
    gvxr.usePointSource();
    gvxr.useParallelBeam();
    for energy, percentage in energy_spectrum_in_keV:
        gvxr.addEnergyBinToSpectrum(energy, "keV", percentage);

    # Set up the detector
    print("Set up the detector");
    gvxr.setDetectorPosition(-distance_object_detector_in_m, 0.0, 0.0, "m");
    gvxr.setDetectorUpVector(0, 1, 0);
    gvxr.setDetectorNumberOfPixels(detector_width_in_pixels, detector_height_in_pixels);
    gvxr.setDetectorPixelSize(pixel_size_in_micrometer, pixel_size_in_micrometer, "micrometer");

    global angular_step;

    angular_step = angular_span_in_degrees / number_of_projections;

    print("Number of projections: ", str(number_of_projections))
    print("angular_span_in_degrees: ", str(angular_span_in_degrees))
    print("angular_step: ", str(angular_step))
   
def getDistance(i, j):
    global g_fly_algorithm;

    x_i = g_fly_algorithm.getIndividual(i).getGene(0);
    y_i = g_fly_algorithm.getIndividual(i).getGene(1);

    x_j = g_fly_algorithm.getIndividual(j).getGene(0);
    y_j = g_fly_algorithm.getIndividual(j).getGene(1);

    x_i = g_matrix_x + (x_i - 0.5) * max(g_matrix_width, g_matrix_height) * 2.0;
    y_i = g_matrix_y + (y_i - 0.5) * max(g_matrix_width, g_matrix_height) * 2.0;

    x_j = g_matrix_x + (x_j - 0.5) * max(g_matrix_width, g_matrix_height) * 2.0;
    y_j = g_matrix_y + (y_j - 0.5) * max(g_matrix_width, g_matrix_height) * 2.0;

    return (math.sqrt((x_i - x_j) * (x_i - x_j) + (y_i - y_j) * (y_i - y_j)));

def killBadFlies():
    g_fly_algorithm.computeGlobalFitness();

    for i in range(g_fly_algorithm.getPopulationSize()):
        
        if (g_fly_algorithm.getLocalFitness(i)):
            g_fly_algorithm.deactivateIndividual(i);
            g_fly_algorithm.computeGlobalFitness();


def cleanBadFlies(aFlag):
    global g_fly_algorithm;
    print ("Cleaning bad flies");

    has_killed = True;

    while has_killed == True:
        has_killed = False;

        worse_fly = g_fly_algorithm.getWorseIndividualId();
        worse_fitness = g_fly_algorithm.getLocalFitness(worse_fly);

        if worse_fitness < 0.0:
            has_killed = True;
            if aFlag:
                g_fly_algorithm.kill(worse_fitness);
            else:
                g_fly_algorithm.deactivateIndividual(worse_fitness);

    print ("Cleaning bad flies: done");


def cleanCloseFlies(aFlag):
    global g_fly_algorithm;
    print ("Cleaning close flies");
    has_killed = True;

    while has_killed == True:
        has_killed = False;
        #print ("Reset cleaning");
        g_fly_algorithm.computePopulationFitnesses();
        local_fitness_set = g_fly_algorithm.getLocalFitnessSet();
        
        for i in range(g_fly_algorithm.getPopulationSize()):
            if g_fly_algorithm.isIndividualActive(i):
                for j in range(g_fly_algorithm.getPopulationSize()):
                    if i != j:
                        if g_fly_algorithm.isIndividualActive(j):

                            distance = getDistance(i, j);
                            #print ("\t\t\t", i, "/", j, "\t", distance, "\t", 2.0 * fiber_radius);

                            if distance <= 2.0 * fiber_radius:
                                fitness_i = local_fitness_set[i];
                                fitness_j = local_fitness_set[j];

                                #print ("\t\t\t\t", fitness_i, "\t", fitness_j)
                                if aFlag:
                                    # Kill i
                                    if fitness_i < fitness_j:
                                        g_fly_algorithm.kill(i);
                                        #has_killed = True;
                                        #print ("Killing");

                                    # Kill j
                                    elif fitness_j < fitness_i:
                                        g_fly_algorithm.kill(j);
                                        #has_killed = True;
                                        #print ("Killing");
                                else:
                                    if fitness_i < fitness_j:
                                        g_fly_algorithm.deactivateIndividual(i);
                                        #has_killed = True;
                                        #print ("Killing");

                                    elif fitness_j < fitness_i:
                                        g_fly_algorithm.deactivateIndividual(j);
                                        #has_killed = True;
                                        #print ("Killing");

                                #setCylinders(g_fly_algorithm.getPopulation(), True);    
                                #gvxr.displayScene();

    print ("Cleaning close flies: done");

def setCylinders(aPopuation, resetAll = True):

    global g_matrix_width;
    global g_matrix_height;
    global g_matrix_x;
    global g_matrix_y;

    # Object

    # Fiber/Core

    if resetAll:
        g_fiber_geometry_set = [];
        g_core_geometry_set  = [];

        gvxr.emptyMesh("g_fiber_geometry");
        gvxr.emptyMesh("g_core_geometry");    
    
        for individual in aPopuation:

            x = g_matrix_x + (individual.getGene(0) - 0.5) * max(g_matrix_width, g_matrix_height) * 2.0;
            y = g_matrix_y + (individual.getGene(1) - 0.5) * max(g_matrix_width, g_matrix_height) * 2.0;
            
            gvxr.makeCylinder("temp_fibre", 100, 815, fiber_radius, "micrometer");
            gvxr.makeCylinder("temp_core", 100, 815,  core_radius, "micrometer");

            gvxr.translateNode("temp_fibre", y, 0, x, "micrometer");
            gvxr.translateNode("temp_core", y, 0, x, "micrometer");
            
            gvxr.addMesh("g_fiber_geometry", "temp_fibre");
            gvxr.addMesh("g_core_geometry",  "temp_core");
            
    gvxr.emptyMesh("fiber_geometry");
    gvxr.emptyMesh("core_geometry");

    gvxr.addMesh("fiber_geometry", "g_fiber_geometry");
    gvxr.addMesh("core_geometry",  "g_core_geometry");

    gvxr.subtractMesh("fiber_geometry", "core_geometry")

    # Fiber
    gvxr.setCompound("fiber_geometry", fiber_material);
    gvxr.setDensity("fiber_geometry", fiber_density, "g.cm-3");

    # Core
    gvxr.setElement("core_geometry", core_material);

    gvxr.removePolygonMeshesFromXRayRenderer();
    gvxr.addPolygonMeshAsOuterSurface("Matrix");
    gvxr.addPolygonMeshAsInnerSurface("fiber_geometry");
    gvxr.addPolygonMeshAsInnerSurface("core_geometry");





def computeSinogram():
# Compute an X-ray image
    #print("Compute sinogram");

    sinogram = np.zeros((number_of_projections, detector_width_in_pixels), dtype=np.float);

    for angle_id in range(0, number_of_projections):
        gvxr.resetSceneTransformation();
        gvxr.rotateScene(-angular_step * angle_id, 0, 1, 0);

        #print (str(angle_id), ":\t", str(angular_step * angle_id), " degrees");
        # Rotate the scene
        
        # Compute the X-ray projection and save the numpy image
        np_image = np.array(gvxr.computeXRayImage());
        
        # Display the 3D scene (no event loop)
        #gvxr.displayScene();
        
        # Append the sinogram
        sinogram[angle_id] = np_image[math.floor(detector_height_in_pixels/2),:];

    total_energy = 0.0;
    for i, j in energy_spectrum_in_keV:
        total_energy += i * j * gvxr.getUnitOfEnergy('keV');


    blur_the_sinogram = False;
    if blur_the_sinogram:
        blurred_sinogram = np.zeros(sinogram.shape);




        t = np.arange(-20., 21., 1.);
        kernel=lsf(t*41)/lsf(0);
        kernel/=kernel.sum();
        #plt.plot(t,kernel);
        #plt.show();


        for i in range(sinogram.shape[0]):
            blurred_sinogram[i]=np.convolve(sinogram[i], kernel, mode='same');

        blurred_sinogram  = total_energy / blurred_sinogram;
        blurred_sinogram  = np.log(blurred_sinogram);
        blurred_sinogram /= (pixel_size_in_micrometer * gvxr.getUnitOfLength("um")) / gvxr.getUnitOfLength("cm");
        
        #np.savetxt("blurred_sinogram_gvxr.txt", blurred_sinogram);
    
        return blurred_sinogram;
        
    # Convert in keV
    sinogram  = total_energy / sinogram;
    sinogram  = np.log(sinogram);
    sinogram /= (pixel_size_in_micrometer * gvxr.getUnitOfLength("um")) / gvxr.getUnitOfLength("cm");

    #np.savetxt("sinogram_gvxr.txt", sinogram);
    
    return sinogram;


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
    
        global g_fly_algorithm
        global g_reference_sinogram
        global g_reference_CT
    
        theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
        reference_CT       = cropCenter(np.loadtxt("W_ML_20keV.tomo-original.txt"),detector_width_in_pixels,detector_width_in_pixels);
        reference_sinogram = radon(reference_CT, theta=theta, circle=True).T

        g_reference_sinogram = normaliseImage(reference_sinogram);
        g_reference_CT       = normaliseImage(reference_CT);
        #g_reference_sinogram = (reference_sinogram - reference_sinogram.mean()) / reference_sinogram.std();
        #g_reference_CT       = (reference_CT       - reference_CT.mean())       / reference_CT.std();


        np.savetxt("sinogram_ref.txt", g_reference_sinogram);
        np.savetxt("CT_ref.txt",       g_reference_CT);
        
        # Create an OpenGL context
        print("Create an OpenGL context")
        gvxr.createWindow();
        gvxr.setWindowSize(512, 512);

        # Create a Fly Algorithm instance
        g_fly_algorithm = initFlyAlgorithm();

        # Create the X-ray simulator
        initXRaySimulator();
        gvxr.enableArtefactFilteringOnCPU();
    
    
    
        self.fig, self.axarr = plt.subplots(2,4);
        
        size_figure = self.fig.get_size_inches();
        size_figure[0] *= 1.75
        size_figure[1] *= 1.25
        
        self.fig.set_size_inches(size_figure, forward=True)
        #self.fig.set_dpi(50)
        
        self.colour_map = plt.cm.jet;

        #self.axarr[0,0].set_title('Reference sinogram')
        #self.axarr[0,1].set_title('Simulated sinogram')
        #self.axarr[0,2].set_title('Absolute error')
        #self.axarr[0,3].set_title('Zero-mean NCC')

        #self.axarr[1,0].set_title('Reference CT')
        #self.axarr[1,1].set_title('Simulated CT')
        #self.axarr[1,2].set_title('Absolute error')
        #self.axarr[1,3].set_title('Zero-mean NCC')
        
        animation.TimedAnimation.__init__(self, self.fig, interval=1, blit=True)
        

    def _draw_frame(self, framedata):
        global g_best_ncc_sinogram;
        global g_best_ncc_CT_slice;
        global g_number_of_generations;
        global g_number_of_mitosis;
        
        # Create a new population
        g_fly_algorithm.evolveGeneration();
        gvxr.displayScene();
        
        setCylinders(g_fly_algorithm.getPopulation(), True);    
        sinogram = computeSinogram();
        gvxr.displayScene();
    
        theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
        test_image = iradon(sinogram.T, theta=theta, circle=True);
    
        normalised_sinogram = normaliseImage(sinogram);
        normalised_CT = normaliseImage(test_image);
        #normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
        #normalised_CT       = (test_image       - test_image.mean())       / test_image.std();

        np.savetxt("before_cleaning_sinogram_gvxr.txt", normalised_sinogram);
        np.savetxt("before_cleaning_CT_gvxr.txt",       normalised_CT);




        cleanCloseFlies(True);
        g_number_of_generations += 1;
        
        # Print the best individual
        #####g_fly_algorithm.getBestIndividual().print()

        # Get its genes
        ####best_individual_s_genes = g_fly_algorithm.getBestIndividual().m_p_gene_set;

        # Reset the geometry using them
        #setCylinders(g_best_population, True);
        setCylinders(g_fly_algorithm.getPopulation(), True);
        gvxr.displayScene();
        
        # Create the corresponding sinogram
        sinogram = computeSinogram();
        
        # Reconstruct it using the FBP algorithm
        theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
        reconstruction_fbp = iradon(sinogram.T, theta=theta, circle=True);
        
        # Normalise the sinogram and the reconstruction
        normalised_sinogram = normaliseImage(sinogram);
        normalised_CT = normaliseImage(reconstruction_fbp);
        #normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
        #normalised_CT       = (reconstruction_fbp - reconstruction_fbp.mean()) / reconstruction_fbp.std()    

        np.savetxt("after_cleaning_sinogram_gvxr.txt", normalised_sinogram);
        np.savetxt("after_cleaning_CT_gvxr.txt",       normalised_CT);

        # Compute the ZNCCs    
        g_best_ncc_sinogram.append(int(100*productImage(normalised_sinogram, g_reference_sinogram).mean()));
        g_best_ncc_CT_slice.append(int(100*productImage(g_reference_CT, normalised_CT).mean()));
        
        print(g_number_of_generations, "\tNCC sinogram: ", g_best_ncc_sinogram[-1], "\tNCC CT slice: ", g_best_ncc_CT_slice[-1]);
        red = (1,0,1,0.5)

        self.ncc_sinogram = self.axarr[0,3].scatter(g_number_of_generations, g_best_ncc_sinogram[-1], color=red, s=2, alpha=1.0)
        self.ncc_CT_slice = self.axarr[1,3].scatter(g_number_of_generations, g_best_ncc_CT_slice[-1], color=red, s=2, alpha=1.0)
        
        if g_number_of_generations == 1:
            self.img = [];
            self.img.append(self.axarr[0,0].imshow(g_reference_sinogram, cmap=self.colour_map))
            self.img.append(self.axarr[0,1].imshow(normalised_sinogram, cmap=self.colour_map))
            self.img.append(self.axarr[0,2].imshow(np.abs(g_reference_sinogram - normalised_sinogram), cmap=self.colour_map))
            
            self.img.append(self.axarr[1,0].imshow(g_reference_CT, cmap=plt.cm.gray))
            self.img.append(self.axarr[1,1].imshow(normalised_CT, cmap=plt.cm.gray))
            self.img.append(self.axarr[1,2].imshow(np.abs(g_reference_CT - normalised_CT), cmap=self.colour_map))

            self.axarr[0,0].set_axis_off();
            self.axarr[0,1].set_axis_off();
            self.axarr[0,2].set_axis_off();
            self.axarr[1,0].set_axis_off();
            self.axarr[1,1].set_axis_off();
            self.axarr[1,2].set_axis_off();
            
            self.axarr[0,0].autoscale(False);
            self.axarr[0,1].autoscale(False);
            self.axarr[0,2].autoscale(False);
            #self.axarr[0,3].autoscale(False);
            
            self.axarr[1,0].autoscale(False);
            self.axarr[1,1].autoscale(False);
            self.axarr[1,2].autoscale(False);
            #self.axarr[1,3].autoscale(False);
            
            self.axarr[0,0].set_title('Reference sinogram')
            self.axarr[0,1].set_title('Simulated sinogram')
            self.axarr[0,2].set_title('Absolute error')
            self.axarr[0,3].set_title('Zero-mean NCC')

            self.axarr[1,0].set_title('Reference CT')
            self.axarr[1,1].set_title('Simulated CT')
            self.axarr[1,2].set_title('Absolute error')
            self.axarr[1,3].set_title('Zero-mean NCC')
            
            self.fig.set_tight_layout(True)

            #print('fig size: {0} DPI, size in inches {1}'.format(self.fig.get_dpi(), self.fig.get_size_inches()))
            #self.axarr[0,3].get_xaxis().set_visible(False)
            #self.axarr[1,3].get_xaxis().set_visible(False)


            
            #self.axarr[0,3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            #self.axarr[1,3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            #self.axarr[0,3].yaxis.set_major_locator(MaxNLocator(integer=True))
            #self.axarr[1,3].yaxis.set_major_locator(MaxNLocator(integer=True))

        else:
            self.img[0].set_data(g_reference_sinogram)
            self.img[1].set_data(normalised_sinogram)
            self.img[2].set_data(np.abs(g_reference_sinogram - normalised_sinogram))
            self.img[3].set_data(g_reference_CT)
            self.img[4].set_data(normalised_CT)
            self.img[5].set_data(np.abs(g_reference_CT - normalised_CT))
            
            self.fig.canvas.draw()
        
        x_tics = np.linspace(0, NUMBER_OF_GENERATIONS * (NUMBER_OF_MITOSIS + 1), NUMBER_OF_GENERATIONS - 1, endpoint=True);
        y_tics = np.linspace(0, 100, 11, endpoint=True);

        self.axarr[0,3].get_xaxis().set_ticks(x_tics);
        self.axarr[1,3].get_xaxis().set_ticks(x_tics);

        self.axarr[0,3].get_yaxis().set_ticks(y_tics);
        self.axarr[1,3].get_yaxis().set_ticks(y_tics);

        self.axarr[0,3].get_xaxis().set_label("Generation #");
        self.axarr[1,3].get_xaxis().set_label("Generation #");

        self.axarr[0,3].get_yaxis().set_label("%");
        self.axarr[1,3].get_yaxis().set_label("%");

        g_fly_algorithm.computePopulationFitnesses();

        good_individuals = sum(1 for item in g_fly_algorithm.getLocalFitnessSet() if item>(0.0))
        print("Good individuals: ", good_individuals, "/", g_fly_algorithm.getPopulationSize())

        # Time for a mitosis
        if not (g_number_of_generations % NUMBER_OF_GENERATIONS):# or good_individuals >= len(g_fly_algorithm.m_p_local_fitness) / 2:
            print ("Mitosis at Generation #", g_number_of_generations, " from ", g_fly_algorithm.getPopulationSize(), " to ", g_fly_algorithm.getPopulationSize() * 2, " individuals")

            g_fly_algorithm.mitosis();
            g_number_of_mitosis += 1;
            
        #line1.set_ydata(y1)
        #draw()
        #for ax in self.axarr:
        #    ax.set_axis_off();
        #    ax.autoscale(False);
        
        
        #self.fig.tight_layout()

    def new_frame_seq(self):
        return iter(range(NUMBER_OF_GENERATIONS * (NUMBER_OF_MITOSIS + 1)))


def setMatrix(apGeneSet):

    # Matrix
    # Make a cube
    w = apGeneSet[2] * detector_width_in_pixels * pixel_size_in_micrometer / 1.5;
    h = apGeneSet[3] * detector_width_in_pixels * pixel_size_in_micrometer / 1.5;

    x = apGeneSet[0] * detector_width_in_pixels * pixel_size_in_micrometer - detector_width_in_pixels * pixel_size_in_micrometer / 2.0;
    y = apGeneSet[1] * detector_width_in_pixels * pixel_size_in_micrometer - detector_width_in_pixels * pixel_size_in_micrometer / 2.0;

    gvxr.makeCube("Matrix", 1, "micrometer");
    gvxr.addPolygonMeshAsInnerSurface("Matrix");        
    gvxr.rotateNode("Matrix", 0, 1, 0, apGeneSet[4] * 360.0);
    gvxr.scaleNode("Matrix", w, 815, h, "mm");
    gvxr.translateNode("Matrix", y, 0, x, "micrometer");
    
    gvxr.setMixture("Matrix", matrix_material);
    gvxr.setDensity("Matrix", matrix_density, "g.cm-3");


   
################################################################################
# Run the script
################################################################################

try:

    # Initialise the X-ray system and optimisation algorithm
    ani = SubplotAnimation()



    #best_individual_s_genes = [0.5859774174670042, 0.3244259243528434, 0.6062948701961279, 0.46512532982794313, 0.36295507347649336];
    #best_individual_s_genes = [0.5862890780780811, 0.330976492941764, 0.6185064030596172, 0.3813500904639961, 1.0901447493220267];
    #best_individual_s_genes = [0.5744659660830413, 0.34122547300761086, 0.5399103790120158, 0.5005289054771243, 0.3716677835584586];
    
    best_individual_s_genes = [0.6088925283126014, 0.33814355531109763, 0.6622725678309, 0.584535159378608, 0.500262379003001]; # NCC = 0.9480978675194939
    
    setMatrix(best_individual_s_genes);
    
    g_matrix_width  = best_individual_s_genes[2] * detector_width_in_pixels * pixel_size_in_micrometer / 1.5;
    g_matrix_height = best_individual_s_genes[3] * detector_width_in_pixels * pixel_size_in_micrometer / 1.5;
    g_matrix_x = best_individual_s_genes[0] * detector_width_in_pixels * pixel_size_in_micrometer - detector_width_in_pixels * pixel_size_in_micrometer / 2.0;
    g_matrix_y = best_individual_s_genes[1] * detector_width_in_pixels * pixel_size_in_micrometer - detector_width_in_pixels * pixel_size_in_micrometer / 2.0;

    #setCylinders(g_fly_algorithm.getPopulation(), True);
    #gvxr.renderLoop();

    #cleanCloseFlies(True);
    setCylinders(g_fly_algorithm.getPopulation(), True);
    
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    #plt.rcParams['animation.ffmpeg_args'] = '-report'
    
    #print( plt.rcParams["savefig.bbox"])           # None
    #print( plt.rcParams["animation.writer"])       # ffmpeg
    #print( plt.rcParams["animation.codec"])        # h264
    #print( plt.rcParams["animation.ffmpeg_path"])  # ffmpeg
    #print( plt.rcParams["animation.ffmpeg_args"])  # []
    
    
    metadata = dict(title='Rectangle registration using gVirtualXRay', artist='Dr. F. P. Vidal', comment='Video created to illustrate the capabilities of gVirtualXRay in my talk at IBFEM-4i.')
    writer = animation.FFMpegWriter(fps=1, codec='rawvideo', metadata=metadata)
    


    # Display the 3D scene (no event loop)
    # Run an interactive loop 
    # (can rotate the 3D scene and zoom-in)
    # Keys are:
    # Q/Escape: to quit the event loop (does not close the window)
    # B: display/hide the X-ray beam
    # W: display the polygon meshes in solid or wireframe
    # N: display the X-ray image in negative or positive
    # H: display/hide the X-ray detector
    # V: display/hide the normal vectors
    gvxr.renderLoop();

    
    




    # Run the animation and save in a file
    ani.save('cylinders_sinogram.gif', fps=1, writer='imagemagick', metadata=metadata);
    
    # Run the animation
    #plt.show()

    g_fly_algorithm.setPopulation(copy.deepcopy(g_best_population));


    setCylinders(g_fly_algorithm.getPopulation(), True);    
    sinogram = computeSinogram();
    gvxr.displayScene();
    
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    test_image = iradon(sinogram.T, theta=theta, circle=True);
    
    normalised_sinogram = normaliseImage(sinogram);
    normalised_CT = normaliseImage(test_image);
    #normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
    #normalised_CT       = (test_image       - test_image.mean())       / test_image.std();

    # Fitness based on NCC
    ncc_sinogram =  productImage(g_reference_sinogram, normalised_sinogram).mean();
    ncc_CT       =  productImage(g_reference_CT, normalised_CT).mean();

    print("Before Killing bad flies \tNCC sinogram: ", 100*ncc_sinogram, "\tNCC CT slice: ", 100*ncc_CT);

    np.savetxt("before_killing_bad_flies_sinogram_gvxr.txt", normalised_sinogram);
    np.savetxt("before_killing_bad_flies_CT_gvxr.txt",       normalised_CT);

    cleanBadFlies(False);

    setCylinders(g_fly_algorithm.getPopulation(), False);    
    sinogram = computeSinogram();
    gvxr.displayScene();
    
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    test_image = iradon(sinogram.T, theta=theta, circle=True);
    
    normalised_sinogram = normaliseImage(sinogram);
    normalised_CT = normaliseImage(test_image);
    #normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
    #normalised_CT       = (test_image       - test_image.mean())       / test_image.std();

    # Fitness based on NCC
    ncc_sinogram =  productImage(g_reference_sinogram, normalised_sinogram).mean();
    ncc_CT       =  productImage(g_reference_CT, normalised_CT).mean();

    print("After Killing bad flies \tNCC sinogram: ", 100*ncc_sinogram, "\tNCC CT slice: ", 100*ncc_CT);

    np.savetxt("after_killing_bad_flies_sinogram_gvxr.txt", normalised_sinogram);
    np.savetxt("after_killing_bad_flies_CT_gvxr.txt",       normalised_CT);




    cleanCloseFlies(False);

    setCylinders(g_fly_algorithm.getPopulation(), False);    
    sinogram = computeSinogram();
    gvxr.displayScene();
    
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    test_image = iradon(sinogram.T, theta=theta, circle=True);
    
    normalised_sinogram = normaliseImage(sinogram);
    normalised_CT = normaliseImage(test_image);
    #normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
    #normalised_CT       = (test_image       - test_image.mean())       / test_image.std();

    normalised_sinogram = normaliseImage(sinogram);
    normalised_CT = normaliseImage(test_image);

    # Fitness based on NCC
    ncc_sinogram =  productImage(g_reference_sinogram, normalised_sinogram).mean();
    ncc_CT       =  productImage(g_reference_CT, normalised_CT).mean();

    print("Final\tNCC sinogram: ", 100*ncc_sinogram, "\tNCC CT slice: ", 100*ncc_CT);
    
    # Display the 3D scene (no event loop)
    gvxr.displayScene();
        
    np.savetxt("final_sinogram_gvxr.txt", normalised_sinogram);
    np.savetxt("final_CT_gvxr.txt",       normalised_CT);

    blurred_sinogram = np.zeros(sinogram.shape);

    t = np.arange(-20., 21., 1.);
    kernel=lsf(t*41)/lsf(0);
    kernel/=kernel.sum();

    for i in range(sinogram.shape[0]):
        blurred_sinogram[i]=np.convolve(sinogram[i], kernel, mode='same');

    test_image = iradon(sinogram.T, theta=theta, circle=True);
    normalised_CT = normaliseImage(test_image);

    np.savetxt("final_lsf_sinogram_gvxr.txt", normalised_sinogram);
    np.savetxt("final_lsf_CT_gvxr.txt",       normalised_CT);

    for ind in g_fly_algorithm.getPopulation():
        if ind.isActive():
            ind.print();

    gvxr.renderLoop();
   
    exit(); 
except OSError as err:
    print("OS error: {0}".format(err))
except ValueError:
    print("Could not convert data.")
except:
    print("Unexpected error:", sys.exc_info()[0])

