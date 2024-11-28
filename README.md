Harris Corner Detector
======================

*******
SUMMARY
*******

This program implements the Harris corner detector (sequential & parallel). This feature detector 
relies on the analysis of the eigenvalues of the auto-correlation matrix.
The algorithm comprises seven steps, including several measures for the 
classification of corners, a generic non-maximum suppression method for 
selecting interest points, and the possibility to obtain corners position with 
subpixel accuracy.


***********
COMPILATION
***********

Required environment: Any unix-like system with a standard compilation
environment (make and C/C++ compilers)

Required libraries: libpng, lipjpeg, libtiff

Compilation instructions: run "make" to produce an executable


*****
USAGE
*****

The program reads an input images, take some parameters and produce a list of
interest points. 

  Harris corner detector:
  'image' is an input image to detect features on.
  -----------------------------------------------
  OPTIONS:
  --------
   -o name  output image with detected corners 
   -f name  write points to file
   -z N     number of scales for filtering out corners
              default value 1
   -s N     choose smoothing: 
              0.precise Gaussian; 1.fast Gaussian; 2.no Gaussian
              default value 1
   -g N     choose gradient: 
              0.central differences; 1.Sobel operator
              default value 0
   -m N     choose measure: 
              0.Harris; 1.Shi-Tomasi; 2.Harmonic Mean
              default value 0
   -k N     Harris' K parameter
              default value 0.060000
   -d N     Gaussian standard deviation for derivation
              default value 1.000000
   -i N     Gaussian standard deviation for integration
              default value 2.500000
   -t N     threshold for eliminating low values
              default value 130
   -q N     strategy for selecting the output corners:
              0.all corners; 1.sort all corners;
              2.N corners; 3.distributed N corners
              default value 0
   -c N     regions for output corners (1x1, 2x2,...NxN):
              default value 3
   -n N     number of output corners
              default value 2000
   -p N     subpixel accuracy
              0.no subpixel; 1.quadratic approximation; 2.quartic interpolation
              default value 1
   -v       switch on verbose mode

*************
LIST OF FILES
*************

Directory src:
--------------
gaussian.cpp	       : Different implementions of a convolution with a Gaussian function
gradient.cpp           : Functions for computing the gradient of an image
harris.cpp             : This is the main program that implements the Harris method
iio.c                  : Functions for reading and writing images 
interpolation.cpp      : Functions for computing corners with sub-pixel accuracy
main.cpp               : Main algorithm to read parameters and write results
zoom.cpp               : Function for zooming out images by a factor of 2
autocorrelation_gpu.cu : parallel computation of autocorrelation matrix
parallel_computation.cu: parallel computation of discrete gaussian on A,B,C for all images at once (3D grid) 

Compilation:
----------------

mkdir build
cd build
cmake ..
make

Execution:
------------
To enable execute for a set of images, please copy and paste the image in data/test_dir_building several times.
sequential : ./build/sequential data/test_dir_building -r -v 
parallel   : ./build/parallel   data/test_dir_building -r -v
