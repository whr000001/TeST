# Test: Temporal-Spatial Separated Transformer for Temporal Action Localization

This is an official implementation in PyTorch of [TeST](https://www.sciencedirect.com/science/article/abs/pii/S0925231224014590).

<p align="center">
<img src="https://github.com/whr000001/TeST/blob/main/imgs/overview.png"   width="600" />
</p>

## Summary
- We propose TEST, which employs three transformer-based architecture variants, to conduct temporal action localization.
- The three transformer-based architectures can effectively improve localization performance and space-time efficiency.
- We propose to integrate the results from multiple feature maps to obtain more comprehensive predictions.
- Extensive experiments on two real-world benchmarks validate the effectiveness and superiority of our proposed TEST.


## Preparation
This repository is based on [AFSD](https://github.com/TencentYoutuResearch/ActionDetection-AFSD), using a similar code structure and environment.

### Environment
- NVIDIA GPU supporting CUDA 9.2
- CUDA 9.2
- Python 3.7
- Pytorch == 1.4.0 (Please make sure that the pytorch version is 1.4.0 due to the module from AFSD)

### Setup
```
conda install pytorch==1.4.0 cudatoolkit=9.2 -c pytorch
cd AFSD
python setup.py develop
pip install einops 
pip install pyyaml
pip install pandas
```

### Data Preparation
You could follow [AFSD](https://github.com/TencentYoutuResearch/ActionDetection-AFSD) to download the needed data.
- THUMOS14 Data
1. Download the RGB and flow npy files
2. Create a 'data' fold
3. Put npy files into corresponding files

- Backbone Parameters
1. Download the RGB and flow backbone parameters
2. Create a 'backbone' fold
3. Put 'rgb_imagenet.pt' and 'flow_imagenet.pt' in the 'backbone'

### File Structure
Make sure the file structure is correct.
```python
├── AFSD  # the AFSD module to obtain boundary features
│   ├── boundary_max_pooling_cuda.cpp
│   ├── boundary_max_pooling_kernel.cu
│   └── setup.py  # install the AFSD module
├── backbones  # the backbone parameters
│   ├── flow_imagenet.pt  # parameters for flow backbone
│   └── rgb_imagenet.pt  # parameters for rgb backbone
├── configs  # the training configs
│   ├── thumos_flow.yaml  # config for flow model
│   └── thumos_rgb.yaml  # config for rgb model
├── data  # THUMOS dataset
│   ├── test_flow_npy
│   ├── test_npy
│   ├── validation_flow_npy
│   └── validation_npy
├── dataset  # dataset files
│   ├── __init__.py
│   ├── dataset.py
│   └── video_transforms.py
├── imgs  # iamges for README
│   └── overview.png
├── model  # the main model files
│   ├── __init__.py
│   ├── boundary_pooling_op.py
│   ├── i3d_backbone.py
│   ├── layers.py
│   └── main.py
├── config.py  # process config
├── evaluate.py  # evaluate TeST
├── LICENSE
├── losses.py  # the loss function
├── README.md
├── test_ensemble.py  # test TeST using single feature map
├── test_single.py  # test TeST using multiple feature map
└── train.py  # train TeST
```

## Training TeST
Train rgb model using:
```
python train.py --config_file configs/thumos_rgb.yaml
```
or train flow model using:
```
python train.py --config_file configs/thumos_flow.yaml
```
You could change the hyper-parameters in corresponding yaml files.

## Testing TeST
You can test TeST using a single feature map by:
```
python test_single.py --config_file configs/thumos_rgb.yaml
python test_single.py --config_file configs/thumos_flow.yaml 
```
or test using multiple feature maps:
```
python test_ensemble.py --config_file configs/thumos_rgb.yaml 
```
## Evaluating TeST
You can evaluate the output by running:
```
python evaluate.py --output_json output/detection_results_ensemble.json
```

## Citation
If you find our work interesting/helpful, please consider citing TeST:
```
@article{wan2024test,
  title={TeST: Temporal-spatial separated transformer for temporal action localization},
  author={Wan, Herun and Luo, Minnan and Li, Zhihui and Wang, Yang},
  journal={Neurocomputing},
  pages={128688},
  year={2024},
  publisher={Elsevier}
}
```


## Updating
- 20241030: We uploaded the complete codes for the THUMOS14 dataset and completed the readme.
- 20241006: We uploaded the initial codes without details. We plan to upload the complete codes and details by November.
