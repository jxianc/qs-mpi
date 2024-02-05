# Parallel Quick Sort with MPI

## What is this project?
This is an implementation of parallel quick sort algorithm with MPI

## Problem
Given a input, an integer `N`, generates `N` 32-bit non-negative integers randomly and saves to a file the generated numbers in non-decreasing order (ascending order). 

## Apporach
Please read [approach](https://github.com/jxianc/qs-mpi/blob/main/approach.pdf).

- Compile
  ```bash
  make qs_mpi
  ```

- Run 
  ```bash
  mpirun -np <p> ./qs_mpi <N> <output>
  ```
  - `p` - the number of processes
  - `N` - N
  - `output` - the output file

## References
- Chapter 9.4 Quicksort from **Introduction to Parallel Computing By Ananth Grama, Anshul Gupta, George Karypis, Vipin Kumar**