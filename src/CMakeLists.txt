ADD_LIBRARY(hcd STATIC gaussian.cpp gradient.cpp harris.cpp harris_parallel.cpp interpolation.cpp zoom.cpp iio.c)
ADD_LIBRARY(hcd_cuda STATIC gradient_gpu.cu autocorrelation_gpu.cu parallel_computation.cu)
