# COMS 4995 Geometric Data Analysis Final Project – Mesh Parameterization

> **To get started:** Clone this repository with
> 
>     git clone --recursive https://github.com/honglin-c/mesh-parameterization.git
>

## Compilation
Starting in this directory, issue:

    mkdir build
    cd build
    cmake ..
    make 

## Execution

Once built, you can execute the program from inside the `build/` by running
on a given mesh:

    ./parameterization [path to mesh with boundary.obj]
