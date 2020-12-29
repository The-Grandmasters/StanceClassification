## Introduction
This project was initially a class project for our Soft Computing class. We created several classification models that identify the martial arts stance of a human subject as 1 of 7 possible stances. We sourced our own dataset for this project using online available images as well as images of our own.
If you'd like to learn more about the project and see our results, please read our report included in this repository (StanceClassificationFinalReport.pdf).
If you'd like to run our models for yourself, please see the section below for instructions on how to run our code. (Work in Progress)
 
## Dataset
* see the VectorClassification/Output to get the dataset the data, currently just labeled via filename. Use ```torch.load('file.pt')```
to get each tensor.

## Environment Setup 
* I began working on linux, haven't tested on windows yet. 
1. Install conda (conda is a python environment/package manager )
    * https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
2. Add the conda environment to you system with the required packages, using the env .yml file. 

    ```conda env create -f environment.yml```
3. You can then activate that environment using 

    ```conda activate poseClass```

4. Get the 3D vectors of a 2D image using the infer_batch.py or infer_single.py files. Infer single takes one image, while infer_batch takes a directory containing images. Both take in a pretrained models created by the original authors. (https://github.com/anibali/margipose)

Their are two models which can be downloaded from the following external links provided by the authors:
margipose-mpi3d.pth: https://cloudstor.aarnet.edu.au/plus/s/fg5CCss8o9PdURs  
margipose-h36m.pth: https://cloudstor.aarnet.edu.au/plus/s/RisOjU8YwqUXFI7  

Running the infer_batch command would look like this:

```python infer_batch.py --model pretrainedModels/margipose-mpi3d.pth --dir ../Data/images/stances --out output ```
