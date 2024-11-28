// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Javier Sánchez Pérez <jsanchez@ulpgc.es>
// All rights reserved.


#include <stdio.h>
#include <math.h>
#include <sys/time.h> 
#include <algorithm>

#include "harris.h"
#include "gradient.h"
#include "gaussian.h"
#include "zoom.h"
#include "interpolation.h"


using namespace std;


/**
  *
  * Overload less than function to compare two corners
  *
**/
bool operator<(
  const harris_corner &c1, 
  const harris_corner &c2
) 
{
  //inverse sort
  return c1.R > c2.R;
}


/**
  *
  * Function for computing the Autocorrelation matrix
  *
**/
void compute_autocorrelation_matrix(
  float *Ix,   //gradient of the image
  float *Iy,   //gradient of the image
  float *A,    //first coefficient of the autocorrelation matrix: G*(IxIx)
  float *B,    //symmetric coefficient of the autocorrelation matrix: G*(IxIy)
  float *C,    //last coefficient of the autocorrelation matrix: G*(IyIy)
  float sigma, //standard deviation for smoothing 
  int   nx,    //number of columns of the image
  int   ny,    //number of rows of the image
  int   gauss  //type of Gaussian 

)
{
  for (int i=0;i<nx*ny;i++)
  {
     A[i] = Ix[i]*Ix[i];
     B[i] = Ix[i]*Iy[i];
     C[i] = Iy[i]*Iy[i];
  }
  

} 


/**
  *
  * Function for computing Harris' discriminant function
  *
**/
void compute_corner_response(
  float *A,      //upper-left coefficient of the Autocorrelation matrix
  float *B,      //symmetric coefficient of the Autocorrelation matrix
  float *C,      //bottom-right coefficient of the Autocorrelation matrix
  float *R,      //corner strength function
  int   measure, //measure strategy
  int   nx,      //number of columns of the image
  int   ny,      //number of rows of the image
  float k        //Harris coefficient for the measure function
)
{
  int size = nx*ny;
  
  //compute the corner strength function following one strategy
  switch(measure) 
  {
    default: case HARRIS_MEASURE:
      //#pragma omp parallel for
      for (int i=0; i<size; i++)
      {
        float detA  =A[i]*C[i]-B[i]*B[i];
        float traceA=A[i]+C[i];

        R[i]=detA-k*traceA*traceA;          
      }
      break;

    case SHI_TOMASI_MEASURE:
      //#pragma omp parallel for
      for (int i=0; i<size; i++)
      {
        float D=sqrt(A[i]*A[i]-2*A[i]*C[i]+4*B[i]*B[i]+C[i]*C[i]);
        float lmin=0.5*(A[i]+C[i])-0.5*D;

        R[i]=lmin;
      }
      break;

    case HARMONIC_MEAN_MEASURE: 
      //#pragma omp parallel for
      for (int i=0; i<size; i++)
      {
        float detA  =A[i]*C[i]-B[i]*B[i];
        float traceA=A[i]+C[i];

        R[i]=2*detA/(traceA+0.0001);
      }
      break;
  }
}


/**
  *
  * Function for non-maximum suppression
  *
**/
void non_maximum_suppression(
  float *R,             //input data
  vector<harris_corner> &corners, //Harris' corners
  float Th,             //threshold for low values
  int   radius,         //window radius
  int   nx,             //number of columns of the image
  int   ny              //number of rows of the image
)
{
  //check parameters
  if(ny<=2*radius+1 || nx<=2*radius+1) return;
  if(radius<1) radius=1;
  
  int *skip = new int[nx*ny];
  
  //skip values under the threshold
  //#pragma omp parallel for
  for(int i=0; i<nx*ny; i++)
    if(R[i]<Th) skip[i]=1;
    else skip[i]=0;

  //use an array for each row to allow parallel processing
  vector<vector<harris_corner> > corners_row(ny-2*radius);
 
  //#pragma omp parallel for
  for(int i=radius; i<ny-radius; i++)
  {
    int j=radius;
    
    //avoid the downhill at the beginning
    while(j<nx-radius && (skip[i*nx+j] || R[i*nx+j-1]>=R[i*nx+j]))
      j++;
      
    while(j<nx-radius)
    {
      //find the next peak 
      while(j<nx-radius && (skip[i*nx+j] || R[i*nx+j+1]>=R[i*nx+j]))
        j++;
      
      if(j<nx-radius)
      {
        int p1=j+2;

        //find a bigger value on the right
        while(p1<=j+radius && R[i*nx+p1]<R[i*nx+j]) 
        {
          skip[i*nx+p1]=1;
          p1++;
        }

        //if not found
        if(p1>j+radius)
        {  
          int p2=j-1;
  
          //find a bigger value on the left
          while(p2>=j-radius && R[i*nx+p2]<=R[i*nx+j])
            p2--;
  
          //if not found, check the 2D region
          if(p2<j-radius)
          {
            int k=i+radius; 
            bool found=false;

            //first check the bottom region (backwards)
            while(!found && k>i)
            {
              int l=j+radius;
              while(!found && l>=j-radius)
              {
                if(R[k*nx+l]>R[i*nx+j])
                  found=true;
                else skip[k*nx+l]=1;
                l--;
              }
              k--;
            }
            
            k=i-radius; 

            //then check the top region (forwards)
            while(!found && k<i)
            {
              int l=j-radius;
              while(!found && l<=j+radius)
              {
                if(R[k*nx+l]>=R[i*nx+j])
                  found=true;
                
                l++;
              }
              k++;
            }
            
            if(!found)
              //a new local maximum detected
              corners_row[i-radius].push_back({(float)j, (float)i, R[i*nx+j]});
          }
        }
        j=p1;
      }
    }
  }
  
  //copy row corners to the output list
  for(int i=0; i<ny-2*radius; i++)
   corners.insert(corners.end(), corners_row[i].begin(), corners_row[i].end());
  
  delete []skip;
}


/**
  *
  * Function for selecting the output corners
  *
**/    
void select_output_corners(
  std::vector<harris_corner> &corners, //output selected corners
  int strategy, //strategy for the output corners
  int cells,    //number of regions in the image for distributed output
  int N,        //number of output corners
  int nx,       //number of columns of the image
  int ny        //number of rows of the image
)
{
  switch(strategy) 
  {
    default: case ALL_CORNERS: 
      break;

    case ALL_CORNERS_SORTED:
      sort(corners.begin(), corners.end());
      break;

    case N_CORNERS:
      sort(corners.begin(), corners.end());
      if(N<(int)corners.size())
        corners.erase(corners.begin()+N, corners.end());
      break;

    case DISTRIBUTED_N_CORNERS:
      int cellx=cells, celly=cells;
      if(cellx>nx) cellx=nx;
      if(celly>ny) celly=ny;

      int size=cellx*celly;
      int Ncell=N/size;
      if(Ncell<1) Ncell=1;
      vector<vector<harris_corner>> cell_corners(size);
            
      float Dx=(float)nx/cellx;
      float Dy=(float)ny/celly;
      
      //distribute corners in the cells
      for(unsigned int i=0; i<corners.size(); i++)
      {
        int px=(float)corners[i].x/Dx;
        int py=(float)corners[i].y/Dy;
        cell_corners[(int)(py*cellx+px)].push_back(corners[i]);
      }

      //sort the corners in each cell
      for(int i=0; i<size; i++)
        sort(cell_corners[i].begin(), cell_corners[i].end());

      //copy the Ncell first corners to the output array
      corners.resize(0);
      for(int i=0; i<size; i++)
        if((int)cell_corners[i].size()>Ncell)
          corners.insert(
            corners.end(), cell_corners[i].begin(), 
            cell_corners[i].begin()+Ncell
          );
        else
          corners.insert(
            corners.end(), cell_corners[i].begin(), 
            cell_corners[i].end()
          );
      
      //sort output corners
      sort(corners.begin(), corners.end());
      if(N<(int)corners.size())
        corners.erase(corners.begin()+N, corners.end());
      break;
  }
}


/**
  *
  * Function for computing subpixel precision of maxima
  *
**/
void compute_subpixel_precision(
  float *R, //discriminant function
  vector<harris_corner> &corners, //selected corners
  int nx,   //number of columns of the image
  int type  //type of interpolation (quadratic or quartic)
)
{
  //#pragma omp parallel for
  for(unsigned int i=0; i<corners.size(); i++)
  {
    int x=corners[i].x;
    int y=corners[i].y;
   
    int mx=x-1;
    int dx=x+1;
    int my=y-1;
    int dy=y+1;
    
    float M[9];
    M[0]=R[my*nx+mx];
    M[1]=R[my*nx+ x];
    M[2]=R[my*nx+dx];
    M[3]=R[y *nx+mx];
    M[4]=R[y *nx+ x];
    M[5]=R[y *nx+dx];
    M[6]=R[dy*nx+mx];
    M[7]=R[dy*nx+ x];
    M[8]=R[dy*nx+dx];
        
    if(type==QUADRATIC_APPROXIMATION)
      quadratic_approximation(
        M, corners[i].x, corners[i].y, corners[i].R
      );
    else if(type==QUARTIC_INTERPOLATION)
      quartic_interpolation(
        M, corners[i].x, corners[i].y, corners[i].R
      );
    
  }   
}


/**
  *
  * Functions for printing messages in verbose mode
  *
**/
void message(const char *msg, timeval &start, int verbose)
{ 
  if (verbose)
  {
     printf("%s", msg);
     gettimeofday(&start, NULL);
  }
}  

void message(const char *msg, timeval &start, timeval &end, int verbose)
{ 
  if (verbose)
  {
     gettimeofday(&end, NULL);
     printf("Time: %fs\n", ((end.tv_sec-start.tv_sec)* 1000000u + 
            end.tv_usec - start.tv_usec) / 1.e6);

     printf("%s", msg);
     gettimeofday(&start, NULL);
  }
}  

void message(timeval &start, timeval &end)
{ 
   gettimeofday(&end, NULL);
   printf("Time: %fs\n", ((end.tv_sec-start.tv_sec)* 1000000u + 
          end.tv_usec - start.tv_usec) / 1.e6);
}  


/**
  *
  * Function for computing the square distance of two corners at 
  * two different scales
  *
**/
float distance2(
  harris_corner &c1, //corner at fine scale
  harris_corner &c2  //corner at coarse scale
)
{
  float dx=(c2.x-c1.x/2.);
  float dy=(c2.y-c1.y/2.);
  
  return dx*dx+dy*dy;
}


/**
  *
  * Function for selecting stable corners comparing 
  * two different scales
  *
**/
void select_corners(
  vector<harris_corner> &corners,   //corners at fine scale
  vector<harris_corner> &corners_z, //corners at coarse scale
  float sigma_i
)
{
  //select stable corners
  vector<harris_corner> corners_t; 
  for(unsigned int i=0; i<corners.size(); i++)
  {
    unsigned int j=0;
    
    //search the corresponding corner
    while(j<corners_z.size() && 
         distance2(corners[i], corners_z[j])>sigma_i*sigma_i) 
      j++;
 
    if(j<corners_z.size())
      corners_t.push_back(corners[i]);
  }
 
  corners.swap(corners_t);
} 


/**
  *
  * Main function for computing Harris corners
  *
**/
void harris(
  float *I,        //input images
  vector<vector<harris_corner>> &corners, //output selected corners
  int   gauss,     //type of Gaussian 
  int   grad,      //type of gradient
  int   measure,   //measure for the discriminant function
  float k,         //Harris constant for the ....function
  float sigma_d,   //standard deviation for smoothing (image denoising)    
  float sigma_i,   //standard deviation for smoothing (pixel neighbourhood)
  float Th,        //threshold for eliminating low values
  int   strategy,  //strategy for the output corners
  int   cells,     //number of regions in the image for distributed output
  int   N,         //number of output corners
  int   precision, //type of subpixel precision approximation
  int   nx,        //number of columns of the image
  int   ny,        //number of rows of the image
  int   verbose,   //activate verbose mode
  int nbr_imgs  // Number of images
)
{
  //check the dimensions of the image
  if(nx<3 || ny<3) return;
  
  struct timeval start, end;
  int size=nx*ny;
  std::vector<float *> Ix;
  std::vector<float *> Iy;
  std::vector<float *> A;
  std::vector<float *> B;
  std::vector<float *> C;
  std::vector<float *> R;

  for (int index_img = 0; index_img < nbr_imgs; index_img++) {
    Ix.push_back(new float[nx*ny]);
    Iy.push_back(new float[nx*ny]);
    A.push_back(new float[nx*ny]);
    B.push_back(new float[nx*ny]);
    C.push_back(new float[nx*ny]);
    R.push_back(new float[nx*ny]);
  }
  // Create a std::vector to store all images
  std::vector<float *> allImages;

  // Iterate through each image
  for (int index_img = 0; index_img < nbr_imgs; index_img++) {
      // Create a temporary vector to store the content of the current image
      float * currentImage = new float[nx*ny];

      // Copy the content of the current image to the temporary vector
      for (int j = 0; j < nx * ny; j++) {
          currentImage[j] = I[index_img * nx * ny + j];
      }

      // Add the temporary vector to the vector containing all images
      allImages.push_back(currentImage);
  }

    verbose=1;
  if(verbose) 
  {
    printf("\nHarris corner detection:\n");
    printf("[nx=%d, ny=%d, sigma_i=%f]\n", nx, ny, sigma_i);
  }
  
  message(" 1.Smoothing the image: \t \t", start, verbose);
  for (int index_img = 0; index_img < nbr_imgs; index_img++){
    gaussian(allImages[index_img], allImages[index_img], nx, ny, sigma_d, gauss);
  }
  
  message(" 2.Computing the gradient: \t \t", start, end, verbose);
  for (int index_img = 0; index_img < nbr_imgs; index_img++){
    gradient(allImages[index_img], Ix[index_img], Iy[index_img], nx, ny, grad);
  }

  message(" 3.a.Computing the autocorrelation sequentially: \t", start, end, verbose);
  for (int index_img = 0; index_img < nbr_imgs; index_img++){
    compute_autocorrelation_matrix(Ix[index_img], Iy[index_img], A[index_img], B[index_img], C[index_img], sigma_i, nx, ny, gauss);
    }

  message(" 3.b.Computing gaussians A, B, C : \t", start, end, verbose);
  gauss=STD_GAUSSIAN;
  for (int index_img =0; index_img < nbr_imgs; index_img++){
    gaussian(A[index_img], A[index_img], nx, ny, sigma_i, gauss);
    //printf("After conv lines & columns A[0:3] : (%f ,%f ,%f ,%f)", A[index_img][0], A[index_img][1], A[index_img][2], A[index_img][3]);
    gaussian(B[index_img], B[index_img], nx, ny, sigma_i, gauss);
    gaussian(C[index_img], C[index_img], nx, ny, sigma_i, gauss);
  }
  message(" 4.Computing corner strength function: \t", start, end, verbose);
  for (int index_img = 0; index_img < nbr_imgs; index_img++){
    compute_corner_response(A[index_img], B[index_img], C[index_img], R[index_img], measure, nx, ny, k);
  }

  message(" 5.Non-maximum suppression:  \t\t", start, end, verbose);
  for (int index_img = 0; index_img < nbr_imgs; index_img++){
    vector<harris_corner> corner;
    non_maximum_suppression(R[index_img], corner, Th, 2*sigma_i+0.5, nx, ny);
    corners.push_back(corner);
  }

  message(" 6.Selecting output corners:  \t\t", start, end, verbose);
  for (int index_img = 0; index_img < nbr_imgs; index_img++){
    select_output_corners(corners[index_img], strategy, cells, N, nx, ny);
  }

  if(precision==QUADRATIC_APPROXIMATION || precision==QUARTIC_INTERPOLATION)
  {
    message(" 7.Calculating subpixel accuracy: \t", start, end, verbose);
    for (int index_img = 0; index_img < nbr_imgs; index_img++){
      compute_subpixel_precision(R[index_img], corners[index_img], nx, precision);
    }
  }
  
  if(verbose)
  {
    message(start, end);
    for (int index_img = 0; index_img < nbr_imgs; index_img++){
      printf(" * Number of corners detected: %ld\n", corners[index_img].size());
    }
  }
  
  for (int index_img = 0; index_img < nbr_imgs; index_img++) {
    delete [] Ix[index_img];
    delete [] Iy[index_img];
    delete [] A[index_img];
    delete [] B[index_img];
    delete [] C[index_img];
    delete [] R[index_img];
  }
}


/**
  *
  * Main function for computing Harris corners with scale test
  *
**/
void harris_scale(
  float *I,        //input images
  vector<vector<harris_corner>> &corners, //output selected corners
  int   Nscales,   //number of scales for checking the stability of corners
  int   gauss,     //type of Gaussian 
  int   grad,      //type of gradient
  int   measure,   //measure for the discriminant function
  float k,         //Harris constant for the ....function
  float sigma_d,   //standard deviation for smoothing (image denoising)    
  float sigma_i,   //standard deviation for smoothing (pixel neighbourhood)
  float Th,        //threshold for eliminating low values
  int   strategy,  //strategy for the output corners
  int   cells,     //number of regions in the image for distributed output
  int   N,         //number of output corners
  int   precision, //type of subpixel precision approximation
  int   nx,        //number of columns of the image
  int   ny,        //number of rows of the image
  int   verbose,   //activate verbose mode
  int   nbr_imgs   // number of images
)
{
  
    if(Nscales<=1 || nx<=64 || ny<=64)
    {

      harris(
        I, corners, gauss, grad, measure, k, sigma_d, sigma_i, 
        Th, strategy, cells, N, precision, nx, ny, verbose, nbr_imgs
      ); 
    }  
    else
    {
      //zoom out the image by a factor of 2
      
      float *Iz = new float[nx*ny*nbr_imgs/4];  
      float* I_img = new float[nx*ny];
      float *Iz_img = new float[nx*ny/4];
      for (int index_img = 0; index_img < nbr_imgs; index_img++){
        for(int j = 0; j < nx*ny; j++){
          I_img[j] = I[index_img*nx*ny+j];
        }

      for(int k = 0; k < nx*ny/4; k++){
          Iz_img = zoom_out(I_img, nx, ny);
          Iz[nx*ny*index_img/4 +k] = Iz_img[k];
        }

  }

     //compute Harris' corners at the coarse scale (recursive)
      vector<vector<harris_corner>> corners_z; 
      harris_scale(
        Iz, corners_z, Nscales-1, gauss, grad, measure, k, sigma_d, 
        sigma_i/2, Th, strategy, cells, N, precision, nx/2, ny/2, verbose, nbr_imgs
      );
      
      delete []Iz;

        
      //compute Harris' corners at the current scale
      harris(
        I, corners, gauss, grad, measure, k, sigma_d, sigma_i, 
        Th, strategy, cells, N, precision, nx, ny, verbose, nbr_imgs
      );
      //select stable corners
      for (int index_img = 0; index_img < nbr_imgs; index_img++){
        select_corners(corners[index_img], corners_z[index_img], sigma_i);
        if(verbose)
          printf(" * Number of corners after scale check: %ld\n", corners[index_img].size());
        }
      printf("fin select cornors\n");
    }
}
