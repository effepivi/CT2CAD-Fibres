#!/usr/bin/env python3
import random
import math
import sys
import os

#from PIL import Image

import numpy as np

# Import the X-ray simulation library
import gvxrPython as gvxr

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


################################################################################
# Global variables
################################################################################

##### Parameters of the Fly Algorithm #####
NUMBER_OF_INDIVIDUALS = 5;
NUMBER_OF_GENES = 2;
NUMBER_OF_GENERATIONS = 25;

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
number_of_projections = 90;
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

g_fly_algorithm1 = 0;
number_of_evaluations1 = 0;
best_ncc1 = 0;

g_best_ncc_sinogram = [];
g_best_ncc_CT_slice = [];


def cropCenter(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
    


################################################################################
# Fitness functions
################################################################################


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


#######################
def initFlyAlgorithm1():
#######################
    
    fly_algorithm = FlyAlgorithm();
    fly_algorithm.setNumberOfIndividuals(NUMBER_OF_INDIVIDUALS, 5);
    Individual.setFitnessFunction(  local_fitness_function1);
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


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
    
        global g_fly_algorithm1
        global g_reference_sinogram
        global g_reference_CT
    
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
        g_fly_algorithm1 = initFlyAlgorithm1();

        # Create the X-ray simulator
        initXRaySimulator();
        gvxr.enableArtefactFilteringOnGPU();
    
    
    
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
        
        self.run_id = 0;

    def _draw_frame(self, framedata):
        global g_best_ncc_sinogram;
        global g_best_ncc_CT_slice;
        global number_of_evaluations1;
        
 
        # Create a new population
        g_fly_algorithm1.evolve()
        self.run_id += 1;
        
        # Print the best individual
        g_fly_algorithm1.getBestIndividual().print()

        # Get its genes
        best_individual_s_genes = g_fly_algorithm1.getBestIndividual().m_p_gene_set;

        # Reset the geometry using them
        setGeometry1(best_individual_s_genes);
        
        # Create the corresponding sinogram
        sinogram = computeSinogram();
        
        # Reconstruct it using the FBP algorithm
        theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
        reconstruction_fbp = iradon(sinogram.T, theta=theta, circle=True);
        
        # Normalise the sinogram and the reconstruction
        normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
        normalised_CT       = (reconstruction_fbp - reconstruction_fbp.mean()) / reconstruction_fbp.std()    

        # Compute the ZNCCs    
        g_best_ncc_sinogram.append(int(100*np.multiply(normalised_sinogram, g_reference_sinogram).mean()));
        g_best_ncc_CT_slice.append(int(100*np.multiply(g_reference_CT, normalised_CT).mean()));
        
        red = (1,0,1,0.5)

        self.ncc_sinogram = self.axarr[0,3].scatter(self.run_id, g_best_ncc_sinogram[-1], color=red, s=2, alpha=1.0)
        self.ncc_CT_slice = self.axarr[1,3].scatter(self.run_id, g_best_ncc_CT_slice[-1], color=red, s=2, alpha=1.0)
        
        if self.run_id == 1:
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

            print('fig size: {0} DPI, size in inches {1}'.format(self.fig.get_dpi(), self.fig.get_size_inches()))
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
           

        self.axarr[0,3].get_yaxis().set_ticks(np.linspace(0, 100, 11, endpoint=True));
        self.axarr[1,3].get_yaxis().set_ticks(np.linspace(0, 100, 11, endpoint=True));
        self.axarr[0,3].get_xaxis().set_ticks(np.linspace(0, NUMBER_OF_GENERATIONS + 1, NUMBER_OF_GENERATIONS + 2, endpoint=True));

        self.axarr[1,3].get_xaxis().set_ticks(np.linspace(0, NUMBER_OF_GENERATIONS + 1, NUMBER_OF_GENERATIONS + 2, endpoint=True));
        
        
        #line1.set_ydata(y1)
        #draw()
        #for ax in self.axarr:
        #    ax.set_axis_off();
        #    ax.autoscale(False);
        
        
        #self.fig.tight_layout()

    def new_frame_seq(self):
        return iter(range(NUMBER_OF_GENERATIONS))


   
################################################################################
# Run the script
################################################################################

#try:
if True:

    # Initialise the X-ray system and optimisation algorithm
    ani = SubplotAnimation()
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    #plt.rcParams['animation.ffmpeg_args'] = '-report'
    
    #print( plt.rcParams["savefig.bbox"])           # None
    #print( plt.rcParams["animation.writer"])       # ffmpeg
    #print( plt.rcParams["animation.codec"])        # h264
    #print( plt.rcParams["animation.ffmpeg_path"])  # ffmpeg
    #print( plt.rcParams["animation.ffmpeg_args"])  # []
    
    
    metadata = dict(title='Rectangle registration using gVirtualXRay', artist='Dr. F. P. Vidal', comment='Video created to illustrate the capabilities of gVirtualXRay in my talk at IBFEM-4i.')
    writer = animation.FFMpegWriter(fps=1, codec='rawvideo', metadata=metadata)
    
    # Run the animation and save in a file
    ani.save('basic_animation.gif', fps=1, writer='imagemagick', metadata=metadata);
    
    # Run the animation
    #plt.show()
    
    exit()
    
    # Print the best individual
    g_fly_algorithm1.getBestIndividual().print()

    # Get its genes
    best_individual_s_genes = g_fly_algorithm1.getBestIndividual().m_p_gene_set;

    # Reset the geometry using them
    setGeometry1(best_individual_s_genes);
    
    # Create the corresponding sinogram
    sinogram = computeSinogram();
    np.savetxt("sinogram_gvxr.txt", sinogram);
        
    # Display the 3D scene (no event loop)
    gvxr.displayScene();

    # Reconstruct it using the FBP algorithm
    theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
    reconstruction_fbp = iradon(sinogram.T, theta=theta, circle=True);
    np.savetxt("reconstruction_gvxr_fbp.txt", reconstruction_fbp);
        
    # Normalise the sinogram and the reconstruction
    normalised_sinogram = (sinogram - sinogram.mean()) / sinogram.std();
    normalised_CT       = (reconstruction_fbp - reconstruction_fbp.mean()) / reconstruction_fbp.std()    ;
    
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

    
    
    
    
    
    
    
    exit();
    
    
        



    anim = animation.FuncAnimation(g_fig, animate, frames=NUMBER_OF_GENERATIONS, blit=True);
    anim.save('basic_animation.avi', fps=30, extra_args=['-vcodec', 'libx264'], writer='ffmpeg');
    plt.show();
    exit();
    
    




#except Exception as error:
#    print ("Exception caught:");
#    exc_type, exc_obj, exc_tb = sys.exc_info()
#    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#    print("\t", exc_type, fname, exc_tb.tb_lineno)

    
exit()


# Scene geometrical properties
# See Figure 2.1 on Page 15 of Modelling the response of X-ray detectors and removing artefacts in 3D tomography

