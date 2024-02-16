# About sobss

Compact building reconstruction based on Single-orientation Building Section Skeleton (BSS). This is an implementation of the [paper](https://doi.org/10.1016/j.isprsjprs.2024.01.020): 

```
Yijie Wu, Fan Xue*, Maosu Li, and Sou-Han Chen, 2024.
A novel Building Section Skeleton for compact 3D reconstruction from point clouds: A study of high-density urban scenes.
ISPRS Journal of Photogrammetry and Remote Sensing 209, 85-100. 
```

# How to use
## Stage 1
### Skeletonization
- Build from source
```
cd 1-skeletonization
mkdir build
cd build
cmake ..
make 
```
- Run with a configuration file
```
./vote [insert configuration path here]
```
### Coarse segmentation
```
cd 2-coarse_segmentation
python main.py 
```
## Stage 2
- Build from source
```
cd 3-merging
mkdir build
cd build
cmake ..
make
```
- Run with a configuration file
```
./merge [insert configuration path here]
```
# Test data and results
Data and results can be downloaded at [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yijiewu_connect_hku_hk/Ej77MEfWVCtLrJVM-27fO40Brt0B5MLBiAErMaX3p0M3YQ?e=2cHrAg): 
- Inputs: 15 scenes tested in the paper can be found in 0-input. 
- Results: Reconstructed results are in 2-union_tri(results). 
- Evaluation: Inputs colorized in terms of the distances to the results are in 3-r2s(evaluation).

# TODO
- [ ] clean the code
- [ ] make the interface more user-friendly  
    - [ ] call the cpp code in python
    - [ ] update the results and tune the parameters in one open3d window (python)

# Contacts

- Wu, Y.: [yijiewu@connect.hku.hk](mailto:yijiewu@connect.hku.hk?subject=[GitHub]sobss)
- Xue, F.: [xuef@hku.hk](mailto:xuef@hku.hk?subject=[GitHub]sobss), [frankxue.com](//frankxue.com/)
