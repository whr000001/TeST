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
3. TBD

- Backbone Parameters
1. Download the RGB and flow backbone parameters
2. Create a 'backbone' fold
3. Put 'rgb_imagenet.pt' and 'flow_imagenet.pt' in the 'backbone'

### File Structure
TBD

## Training TeST
TBD
## Testing TeST
TBD
## Evaluating TeST
TBD

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
