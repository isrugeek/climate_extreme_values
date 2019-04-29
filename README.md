
# Forecasting Extreme values in Time Series for Climate Change
[Project](https://github.com/isrugeek/climate_extreme_values) | [Arxiv](https://arxiv.org/abs/) |
[Dataset](https://kaggle.com/c/short-term-load-forecasting-challenge/data)

Pytorch and Keras implementation for systematic comparison of exist-ing approaches on a pair common tasks, subsea-sonal forecasting and power consumption predic-tion. 

Forecasting Extreme values in Time Series for Climate Change
 [Israel Goytom](http://isrugeek.github.io), [Kris Sankaran](.),[Yoshua Benjio](.)


**Note**: Please Download the dataset file from [Kaggle](https://kaggle.com/c/short-term-load-forecasting-challenge/data) or 
          [Subseasonal Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IHBANG)

## Setup

### Prerequisites
- Linux or OSX
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)
- Pyro
- Pytorch
- Keras with tensorflow backend



### Getting Started
- Install Pytorch visit https://pytorch.org/, Pyro visit http://pyro.ai/
- Install keras with tensorflow backend and dependencies from https://keras.io/#installation
- Install python packages `jupyter-notebook`, `matplotlib`,`scikit` and `pandas` 
```bash
pip install package-name
```
- Install [livelossplot](https://github.com/stared/livelossplot)(optional) - a live monitor during training.

- Clone this repo:
```bash
git clone https://github.com/isrugeek/climate_extreme_values
cd climate_extreme_values
```
Make sure to follow [This](https://github.com/paulo-o/forecast_rodeo)  for preparation for generating forecasts. 

Run the code from the main directory which the README.md is located
