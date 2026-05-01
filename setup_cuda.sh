#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.8  # Change this to your actual path
export CUDA_INCLUDE=$CUDA_HOME/include
export CUDA_LIB=$CUDA_HOME/lib64       # Note: Usually lib64 on modern Linux for CUDA
export LD_LIBRARY_PATH=$CUDA_LIB:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_LIB:$LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
export CXXFLAGS="-I$CUDA_HOME/include $CXXFLAGS"
export CPATH="$CUDA_HOME/include:$CPATH"