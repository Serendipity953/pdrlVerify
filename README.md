# Guide

This repository provides the source code from the paper, the dockerfile used to build the docker image directly to run it, and the raw data from the paper.

In this case, the dockerfile and the raw data are stored in the Artifact folder, while the other folders contain the source code, which should be refactored as the ***pdrlVerify*** folder described in the dockerfile for use.

If you want to run experiments and refactoring directly on the source code, the specific experimental environment can also be configured according to the dockerfile.

For detailed steps on how to run the experiment, please refer to the readme file in artifact.

Due to the number and size of files required to build the image, we do not provide the files for the build in this repository, but the detailed build steps have been described in the dockerfile. The main environment configuration lies in the installation of stormpy and the installation of the anaconda environment, both of which have been described in detail. For the installation of stormpy, if you are confused, you can refer to: https://www.stormchecker.org/documentation/obtain-storm/build.html