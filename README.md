# Virtual Try-on Clothes
This repository uses  computer vision (object segmentation,deep learning) technique to take 2D image of target human and cloth as input to provide a 3D view of the fitting of the cloth selected virtually. 

![Images](https://user-images.githubusercontent.com/65164450/210137423-f96df1f6-d25d-428c-b970-b1c0324ce487.jpg)

## Installation 
Clone this repository:

```
git clone https://github.com/A0158/Try-on.git
cd ./Try-on/
```
Install dependencies:

```
pip install requirements.txt
```
## Running Inference
```
python test.py --name model --dataroot try-on_example --results_dir results
```
## Getting coloured cloud/ mesh output
```
python meshing.py
```
Now you get the point cloud file ready for Meshing.
Open the point cloud in Meshlab
### Normal Estimation:
1) go to Filters -> Normals, Curvatures 
2) Orientation -> Compute normals for point sets
### Poisson Remeshing:
1) Filters --> Remeshing, Simplification 
2) Reconstruction --> Surface Reconstruction (reconstrution depth=9)
