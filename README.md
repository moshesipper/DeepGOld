# Deep GOld

Code accompanying the paper: [M. Sipper, "Combining Deep Learning with Good Old-Fashioned Machine Learning", *SN Computer Science, 2022*](https://arxiv.org/abs/2207.03757).

* `retrain.py`: Retrain a pretrained PyTorch network (Algorithm 1 in the paper).
* `networks.py`: Reconfigure a pretrained network so it can handle a particular dataset (essentially, adjust initial convolutional layer and final fully connected layer).
* `datasets.py`: Definitions (transformations, loading, etc') for the datasets used in the paper. Note the `root` param that the `lambda root, train` of each dataset has; needs to be passed, or -- for ImageNet and Tiny ImageNet -- replaced by your own directory for these datasets.
* `generate_ds.py`: Generate datasets from network outputs, to be used by ML algorithms (first `for` loop, lines 1-8, of Algorithm 2 in paper). Assumes retrained models are in a directory named `Area51`.
* `run_ml.py`: Run ML algorithms on datasets generated from network outputs (second `for` loop, lines 9-15, of Algorithm 2 in paper).
* `latex_table.py`: Create the main latex table of results for the paper.


### Citation

Citations are always appreciated 😊:
```
@Article{Sipper2022Deep,
author={Sipper, Moshe},
title={Combining Deep Learning with Good Old-Fashioned Machine Learning},
journal={SN Computer Science},
year={2022},
month={Dec},
day={08},
volume={4},
number={1},
pages={85},
}
```
