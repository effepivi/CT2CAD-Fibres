#!/usr/bin/env python3
import random
import math

#from PIL import Image

import numpy as np

# Import the X-ray simulation library
import gvxrPython as gvxr

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.transform import radon, iradon, iradon_sart

from lsf import *

from FlyAlgorithm import *


################################################################################
# Global variables
################################################################################

##### Parameters of the Fly Algorithm #####
NUMBER_OF_INDIVIDUALS = 30;
NUMBER_OF_GENES = 2;
NUMBER_OF_GENERATIONS = 5;

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
fiber_material = [("Si", 0.5), ("C", 0.5)];
fiber_mu = 2.736; # cm-1
fiber_density = 3.2; # g/cm3

core_radius = 30 / 2; # um
core_material = [("W", 1)];
core_mu = 341.61; # cm-1
core_density = 19.3 # g/cm3


g_matrix_width = 0;
g_matrix_height = 0;
g_matrix_x = 0;
g_matrix_y = 0;
matrix_material = [("Ti", 0.9), ("Al", 0.06), ("V", 0.04)];
matrix_mu = 13.1274; # cm-1
matrix_density = 4.42 # g/cm3

g_fiber_geometry  = gvxr.emptyMesh();
g_core_geometry   = gvxr.emptyMesh();
g_matrix_geometry = gvxr.emptyMesh();

g_reference_CT       = np.zeros(1);
g_reference_sinogram = np.zeros(1);

g_fiber_geometry_set = [];
g_core_geometry_set  = [];

g_best_population = [];

def cropCenter(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
    
    

################################################################################
# Fitness functions
################################################################################

######################
def rescaleGene(i, j):
######################
    
    if i == 0:
        return j * MAX_VALUE;
    elif i == 1:
        return j * MAX_VALUE;
    else:
        return j * MAX_VALUE;


#########################################################
def local_fitness_function2(ind_id, genes, aFlyAlgorithm):
#########################################################
    
    local_fitness = 0.0;
    global_fitness = aFlyAlgorithm.getGlobalFitness();
    #print("global fitness:\t", global_fitness)

    
    state = aFlyAlgorithm.m_p_population[ind_id].m_is_active;
    
    if state:
        aFlyAlgorithm.m_p_population[ind_id].m_is_active = False;
        local_fitness = global_fitness - global_fitness_function2(aFlyAlgorithm.m_p_population, False);
        aFlyAlgorithm.m_p_population[ind_id].m_is_active = True;
    
    #print("local fitness:\t", local_fitness)
    return local_fitness;


def getNCC(a, b):
    return ((np.multiply(a - a.mean(), b - b.mean()).sum()) / (a.std() * b.std()) / (a.shape[0] * a.shape[1]))

best_ncc1 = 0;
number_of_evaluations1 = 0;

best_ncc2 = 0;
number_of_evaluations2 = 0;


#########################################################
def local_fitness_function1(ind_id, genes, aFlyAlgorithm):
#########################################################
    global best_ncc1;
    global number_of_evaluations1;
    
    setGeometry1(genes);
    sinogram = computeSinogram();
    
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    test_image = iradon(sinogram.T, theta=theta, circle=True);
    
    # Display the 3D scene (no event loop)
    gvxr.displayScene();

    normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
    normalised_CT       = (test_image       - test_image.mean())       / test_image.std();


    number_of_evaluations1 += 1;
    
    # Fitness based on NCC
    ncc =  np.multiply(g_reference_sinogram, normalised_sinogram).mean();

    if best_ncc1 < ncc:
        np.savetxt("sinogram_gvxr.txt", normalised_sinogram);
        np.savetxt("CT_gvxr.txt",       normalised_CT);
        best_ncc1 = ncc;
        #print(number_of_evaluations, ncc*100, genes)
    return (ncc);
    
    
###########################################
def global_fitness_function2(aPopulation, resetAll = True):
###########################################
    
    global best_ncc2;
    global number_of_evaluations2;
    global g_best_population;

    setGeometry2(aPopulation, resetAll);    
    sinogram = computeSinogram();
    
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    test_image = iradon(sinogram.T, theta=theta, circle=True);
    
    # Display the 3D scene (no event loop)
    gvxr.displayScene();

    normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
    normalised_CT       = (test_image       - test_image.mean())       / test_image.std();

    number_of_evaluations2 += 1;
    
    # Fitness based on NCC
    ncc =  np.multiply(g_reference_sinogram, normalised_sinogram).mean();

    if best_ncc2 < ncc:
        np.savetxt("sinogram_gvxr.txt", normalised_sinogram);
        np.savetxt("CT_gvxr.txt",       normalised_CT);
        best_ncc2 = ncc;
        #print(number_of_evaluations, ncc*100, genes)
        g_best_population = copy.deepcopy(aPopulation);

    
    # Fitness based on NCC
    return (ncc);


#######################
def initFlyAlgorithm1():
#######################
    
    fly_algorithm = FlyAlgorithm();
    fly_algorithm.setNumberOfIndividuals(NUMBER_OF_INDIVIDUALS, 5);
    Individual.setFitnessFunction(  local_fitness_function1);
    return (fly_algorithm);
    
    
########################
def initFlyAlgorithm2():
########################
    
    fly_algorithm = FlyAlgorithm();
    fly_algorithm.setNumberOfIndividuals(NUMBER_OF_INDIVIDUALS, NUMBER_OF_GENES);
    fly_algorithm.setLocalFitnessFunction(  local_fitness_function2);
    fly_algorithm.setGlobalFitnessFunction(global_fitness_function2);
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
    

def setGeometry1(apGeneSet):

    global g_matrix_geometry;
   
    # Matrix
    # Make a cube
    w = apGeneSet[2] * detector_width_in_pixels * pixel_size_in_micrometer / 1.5;
    h = apGeneSet[3] * detector_width_in_pixels * pixel_size_in_micrometer / 1.5;
    g_matrix_geometry = gvxr.makeCube(1, "micrometer");
    g_matrix_geometry.rotate(0, 1, 0, apGeneSet[4] * 360.0);
    g_matrix_geometry.scale(w, 815, h);

    x = apGeneSet[0] * detector_width_in_pixels * pixel_size_in_micrometer - detector_width_in_pixels * pixel_size_in_micrometer / 2.0;
    y = apGeneSet[1] * detector_width_in_pixels * pixel_size_in_micrometer - detector_width_in_pixels * pixel_size_in_micrometer / 2.0;

    g_matrix_geometry.translate(y, 0, x, "micrometer");
    
    #print(apGeneSet);
    #print(x, y, w, h, apGeneSet[4])
    #print();
    
    # Matrix
    #temp1.setMaterial(matrix_material);
    #temp1.setDensity(matrix_density, "g.cm-3");
    g_matrix_geometry.setLinearAttenuationCoefficient(matrix_mu, "cm-1");

    gvxr.removePolygonMeshes();
    gvxr.addPolygonMeshAsInnerSurface(g_matrix_geometry, "Matrix");



def setGeometry2(aPopuation, resetAll = True):
    global g_fiber_geometry;
    global g_core_geometry;
    global g_matrix_geometry;
    global g_fiber_geometry_set;
    global g_core_geometry_set;
   
    global g_matrix_width;
    global g_matrix_height;
    global g_matrix_x;
    global g_matrix_y;

    # Object

    # Fiber/Core

    if resetAll:
        g_fiber_geometry_set = [];
        g_core_geometry_set  = [];
    
        for individual in aPopuation:

            g_fiber_geometry_set.append(gvxr.makeCylinder(100, 815, fiber_radius, "micrometer"))
            g_core_geometry_set.append( gvxr.makeCylinder(100, 815,  core_radius, "micrometer"))
            x = g_matrix_x + (individual.getGene(0) - 0.5) * max(g_matrix_width, g_matrix_height) * 2.0;
            y = g_matrix_y + (individual.getGene(1) - 0.5) * max(g_matrix_width, g_matrix_height) * 2.0;

            g_fiber_geometry_set[-1].translate(y, 0, x, "micrometer");
            g_core_geometry_set[ -1].translate(y, 0, x, "micrometer");    
    
    fiber_geometry_set = [];
    core_geometry_set  = [];
    
    for i in range(len(aPopuation)):
    
        if aPopuation[i].m_is_active:
            fiber_geometry_set.append(g_fiber_geometry_set[i])
            core_geometry_set.append(g_core_geometry_set[i])




    # Add the geometries
    fiber_geometry = gvxr.emptyMesh();
    g_core_geometry = gvxr.emptyMesh();

    for fiber in fiber_geometry_set:
        fiber_geometry += fiber;

    for core in core_geometry_set:
        g_core_geometry += core;

    #fiber_geometry.saveSTLFile("fiber.stl");
    #core_geometry.saveSTLFile("core.stl");

    #g_matrix_geometry = matrix_geometry - fiber_geometry;
    g_fiber_geometry  = fiber_geometry  - g_core_geometry;

    # Matrix
    #temp1.setMaterial(matrix_material);
    #temp1.setDensity(matrix_density, "g.cm-3");
    #g_matrix_geometry.setLinearAttenuationCoefficient(matrix_mu, "cm-1");

    # Fiber
    #temp2.setMaterial(fiber_material);
    #temp2.setDensity(fiber_density, "g.cm-3");
    g_fiber_geometry.setLinearAttenuationCoefficient(fiber_mu, "cm-1");

    # Core
    #core_geometry.setMaterial(core_material);
    #core_geometry.setDensity(core_density, "g.cm-3");
    g_core_geometry.setLinearAttenuationCoefficient(core_mu, "cm-1");

    debug = False
    if debug:
        print()
        print ("Material: Ti90/Al6/V4");
        print("Density:", str(g_matrix_geometry.getDensity()), "gm.cm-3")
        print("Mass attenuation coefficient at 33keV:  ", 
            str(g_matrix_geometry.getMassAttenuationCoefficient(33.0, "keV")),
            "cm2.g-1");
        print("Linear attenuation coefficient at 33keV:", str(g_matrix_geometry.getLinearAttenuationCoefficient(33.0, "keV")), "cm-1");


        print()
        print ("Material: SiC");
        print("Density:", str(g_fiber_geometry.getDensity()), "gm.cm-3")
        print("Mass attenuation coefficient at 33keV:  ", 
            str(g_fiber_geometry.getMassAttenuationCoefficient(33.0, "keV")),
            "cm2.g-1");
        print("Linear attenuation coefficient at 33keV:", str(g_fiber_geometry.getLinearAttenuationCoefficient(33.0, "keV")), "cm-1");


        print()
        print ("Material: W");
        print("Density:", str(core_geometry.getDensity()), "gm.cm-3")
        print("Mass attenuation coefficient at 33keV:  ", 
            str(core_geometry.getMassAttenuationCoefficient(33.0, "keV")),
            "cm2.g-1");
        print("Linear attenuation coefficient at 33keV:", str(core_geometry.getLinearAttenuationCoefficient(33.0, "keV")), "cm-1");

    gvxr.removePolygonMeshes();
    gvxr.addPolygonMeshAsOuterSurface(g_matrix_geometry, "Matrix");
    gvxr.addPolygonMeshAsInnerSurface(g_fiber_geometry, "Fiber");
    gvxr.addPolygonMeshAsInnerSurface(g_core_geometry, "Core");



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
        np_image = gvxr.computeXRayImage();
        
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


################################################################################
# Run the script
################################################################################

try:
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    reference_CT       = cropCenter(np.loadtxt("W_ML_20keV.tomo-original.txt"),detector_width_in_pixels,detector_width_in_pixels);
    reference_sinogram = radon(reference_CT, theta=theta, circle=True).T


    g_reference_sinogram = (reference_sinogram - reference_sinogram.mean()) / reference_sinogram.std();
    g_reference_CT       = (reference_CT       - reference_CT.mean())       / reference_CT.std();


    np.savetxt("sinogram_ref.txt", g_reference_sinogram);
    np.savetxt("CT_ref.txt",       g_reference_CT);

    # Create an OpenGL context
    print("Create an OpenGL context")
    gvxr.createWindow();
    gvxr.setWindowSize(512, 512);
    
    # Create a Fly Algorithm instance
    fly_algorithm1 = initFlyAlgorithm1();
    
    # Create the X-ray simulator
    initXRaySimulator();
    gvxr.enableArtefactFilteringOnGPU();

    #computeSinogram();
    
    
    # Compute the X-ray projection and save the numpy image
    #np_image = gvxr.computeXRayImage();
    #gvxr.renderLoop();
    #exit();
    
    # Evolve N times
    #for i in range(NUMBER_OF_GENERATIONS):
    #    fly_algorithm1.evolve()
    #    fly_algorithm1.getBestIndividual().print()
    
    # Get the best match for the matrix
    #best_individual_s_genes = fly_algorithm1.getBestIndividual().m_p_gene_set;
    best_individual_s_genes = [0.5859774174670042, 0.3244259243528434, 0.6062948701961279, 0.46512532982794313, 0.36295507347649336];
    setGeometry1(best_individual_s_genes);

    g_matrix_width  = best_individual_s_genes[2] * detector_width_in_pixels * pixel_size_in_micrometer / 1.5;
    g_matrix_height = best_individual_s_genes[3] * detector_width_in_pixels * pixel_size_in_micrometer / 1.5;
    g_matrix_x = best_individual_s_genes[0] * detector_width_in_pixels * pixel_size_in_micrometer - detector_width_in_pixels * pixel_size_in_micrometer / 2.0;
    g_matrix_y = best_individual_s_genes[1] * detector_width_in_pixels * pixel_size_in_micrometer - detector_width_in_pixels * pixel_size_in_micrometer / 2.0;




    # Create a Fly Algorithm instance
    fly_algorithm2 = initFlyAlgorithm2();
    
    # Create the X-ray simulator
    setGeometry2(fly_algorithm2.m_p_population);
    #computeSinogram();
    
    
    # Compute the X-ray projection and save the numpy image
    #np_image = gvxr.computeXRayImage();
    #gvxr.renderLoop();
    #exit();
    

    fly_algorithm2.m_mutation_probability = 0.0;
    fly_algorithm2.m_crossover_probability = 0.0;
    fly_algorithm2.m_new_blood_probability = 0.20;
    fly_algorithm2.m_elitism_probability = 0.10;

    # Evolve N times
    fly_algorithm2.evolve(NUMBER_OF_GENERATIONS);
    
    fly_algorithm2.m_p_population = copy.deepcopy(g_best_population);
    for i in range(len(fly_algorithm2.m_p_population)):
        if fly_algorithm2.getIndividual(i).computeFitness() < 0:
            fly_algorithm2.getIndividual(i).m_is_active = False;
            fly_algorithm2.computeGlobalFitness();

    detector_width_in_pixels = 1725;
    number_of_projections = 900;
    initXRaySimulator();
    gvxr.enableArtefactFilteringOnCPU();
    setGeometry2(fly_algorithm2.m_p_population);
    sinogram = computeSinogram();
    np.savetxt("sinogram_gvxr.txt", sinogram);
    
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    reconstruction_fbp = iradon(sinogram.T, theta=theta, circle=True);
    np.savetxt("reconstruction_gvxr_fbp.txt", reconstruction_fbp);
    
    
    #reconstruction_sart = iradon(sinogram.T, theta=theta);
    #np.savetxt("reconstruction_gvxr_sart.txt", reconstruction_sart);
    #sinogram *= number_of_projections;

    
    #fly_algorithm.computePopulationFitnesses();

    #for i in range(fly_algorithm.getPopulationSize()):
    #    print(fly_algorithm.getIndividual(i).m_id)
    #    fly_algorithm.getIndividual(i).print();
    
    #print()
    #print(fly_algorithm.m_sort_index)
    
    #print()
    #print(fly_algorithm.m_p_local_fitness)
        
    #print()

    #best = fly_algorithm.getBestIndividualId();
    #print(best)
    #fly_algorithm.getIndividual(best).print()
    #for i in range(Individual.getNumberOfGenes()):
    #    print(rescaleGene(i, fly_algorithm.getIndividual(best).getGene(i)));

    #print()
    #worse = fly_algorithm.getWorseIndividualId();
    #print(worse)
    #fly_algorithm.getIndividual(worse).print()
    #for i in range(Individual.getNumberOfGenes()):
    #    print(rescaleGene(i, fly_algorithm.getIndividual(worse).getGene(i)));



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
except Exception as error:
    print ("Exception caught:");
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print("\t", exc_type, fname, exc_tb.tb_lineno)

    
exit()


# Scene geometrical properties
# See Figure 2.1 on Page 15 of Modelling the response of X-ray detectors and removing artefacts in 3D tomography






def pixel2Cartesian(i):
    return (i - detector_width_in_pixels / 2) * pixel_size_in_micrometer * gvxr.getUnitOfLength('um')






    

#imgplot = plt.imshow((sinogram - sinogram.min() / (sinogram.max() - sinogram.min())), cmap="hot")
#plt.show()

    
# Save the last image into a file
#print("Save the last image into a file");
#gvxr.saveLastXRayImage();



# Normalise the image between 0 and 255 (this is for PIL)
#np_normalised_image = (255 * (np_image-np_image.min())/np_image.max()).astype(np.int8);

# Convert the Numpy array into a PIL image
#img = Image.fromarray(np_normalised_image.astype(np.ubyte));

# Save the PIL image
#img.save('my.png')

# Show the image
#img.show()


# Display the 3D scene (no event loop)
#gvxr.displayScene();


# Display the 3D scene (no event loop)
# Run an interactive loop 
# (can rotate the 3D scene and zoom-in)
# Keys are:
# Q/Escape: to quit the event loop (does not close the window)
# B: display/hide the X-ray beam
# W: display the polygon meshes in solid or wireframe
# N: display the X-ray image in negative or positive
# H: display/hide the X-ray detector
#gvxr.renderLoop();


exit();

