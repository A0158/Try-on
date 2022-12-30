# Virtual Try-on Clothes

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
