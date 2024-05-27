# Connecting NeRFs, Images, and Text
Official code for the paper *Connecting NeRFs, Images, and Text*, accepted at CVPRW 2024. 

[[Paper](https://arxiv.org/abs/2404.07993) | [Project page](https://cvlab-unibo.github.io/clip2nerf) | [Workshop page](https://inrv.github.io)]

## Setup

The code contained in this repository has been tested on Ubuntu 20.04 with Python 3.9.13. 

To install the required packages, run:
```shell
$ sh requirements.sh
```

## Experiments

The [classification](classification/), [retrieval](retrieval/), and [generation](generation/) directories contain the scripts you need to run in order to replicate our results and a README with some instructions on how to do it.

## Datasets

The [dataset](dataset/) directory contains the code to create the embedding datasets used in our experiments. If you need access to the original image/text datasets or to the embedding datasets we used, please contact [francesco.ballerini4@unibo.it](mailto:francesco.ballerini4@unibo.it).

## Cite us

If you find our work useful, please cite us:
```bibtex
@inproceedings{ballerini2024clip2nerf,
    title = {Connecting {NeRFs}, Images, and Text},
    author = {Ballerini, Francesco 
              and Zama Ramirez, Pierluigi 
              and Mirabella, Roberto  
              and Salti, Samuele
              and Di Stefano, Luigi},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year = {2024}
}
```
