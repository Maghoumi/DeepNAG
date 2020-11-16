# DeepNAG: Deep Non-Adversarial Gesture Generation

Official PyTorch implementation of [DeepNAG](https://arxiv.org/abs/2011.09149).

DeepNAG is a novel RNN-based sequence generator that can reliably create synthetic 2D/3D gestures.
Instead of relying on generative adversarial networks (GAN) to train a sequence generator, DeepNAG uses a standalone loss function
based on [soft dynamic time warping](https://github.com/mblondel/soft-dtw) (sDTW) and [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance).
Our novel loss function is intuitive, runs fast and yields great results. Please see [our publication](https://arxiv.org/abs/2011.09149) for more details.
 
DeepNAG's architecture is fairly simple, and only consists of gated recurrent units (GRU) and fully-connected layers: 

<p align="center">
  <img width="500" src="https://github.com/Maghoumi/DeepNAG/blob/master/images/DeepNAG.png"/>
</p>

## Sample Generation Results

### Kinect Gestures

The following is some sample synthetic gestures from the [JK2017 (Kinect)](https://github.com/ISUE/Jackknife/tree/master/datasets/jk2017/kinect) dataset.
In both animations, the black skeleton is an actual person while the remaining skeletons are synthetic!  

<p align="center">
  <img width="400" src="https://github.com/Maghoumi/DeepNAG/raw/master/images/kick.gif"/>
  <img width="400" src="https://github.com/Maghoumi/DeepNAG/raw/master/images/uppercut.gif"/>
</p> 

### Pen Gestures
The following is some sample synthetic gestures from the [$1-GDS](http://depts.washington.edu/acelab/proj/dollar/index.html) dataset.
The red samples are human drawn, while the black samples are synthetic. The last two rows are the overlayed renderings of some randomly
selected real and synthetic samples to demonstrate the diversity of the generated samples compared to the real ones.  

<p align="center">
  <img width="500" src="https://github.com/Maghoumi/DeepNAG/raw/master/images/dollar-gds.png"/>
</p> 

## Getting Started

### Prerequisites

The following is the list of requirements for this project:
- Python v3.6+
- [PyTorch v1.4+](https://pytorch.org/)
- [Numpy](https://numpy.org/) (will be installed along PyTorch)
- [Matplotlib](https://matplotlib.org/) (for visualization)
- [pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) (included as a git submodule)
- [Numba](http://numba.pydata.org/) (preferably install via your OS's package manager)
- (Optional) [Tensorboard](https://www.tensorflow.org/tensorboard) to monitor the training process 
- (Optional) [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) with an NVIDIA GPU (for faster training, although CPU-only training still works, but is very slow)

All the requirements are included in `requirements.txt`.

### Training a Model From Scratch:

Training a model from scratch involves 3 easy steps:

1) Obtain the sources and initialize git submodules
```
git clone https://github.com/Maghoumi/DeepNAG.git
cd DeepNAG
git submodule update --init --recursive
```

2) Install the dependencies (make sure the correct `pip` is used)
```
pip install -r requirements.txt
```

3) Run the code (make sure the correct `python` v3.6+ is used)
```
python main.py --dataset=dollar-gds --use-tensorboard=1
```

The above code will download the [$1-GDS](http://depts.washington.edu/acelab/proj/dollar/index.html) dataset and train the generator on the entire dataset. The dataset will be downloaded to `DeepNAG/data` and the run results will be dumped under `DeepNAG/logs/unique-session-name`.
The training progress will be showed in the standard output. The training progress will additionally be written to tensorboard event files under `DeepNAG/logs/unique-session-name/tensorboard`. By default, training will run for 25000 epochs, which is enough to get good results on the $1-GDS dataset.

The [JK2017 (Kinect)](https://github.com/ISUE/Jackknife/tree/master/datasets/jk2017/kinect) dataset is included in this repository.
To train a model on this dataset, run

```
python main.py --dataset=jk2017-kinect
```

Training will need to run for at least 5000 epochs to produce good results.

### Evaluating a Trained Model:

Once training concludes, the trained model will be saved under `DeepNAG/logs/unique-session-name/checkpoints`. 
The trained model can be visualized using the argument `--evaluate` passed to `main.py`. Some pretrained models are 
included under `DeepNAG/pretrained_models`. Run either of the following commands to visualize a trained model's output (needs [Matplotlib](https://matplotlib.org/)): 

```
# Visualize the pretrained $1-GDS model
python main.py --dataset=dollar-gds --evaluate=pretrained_models/dollar-gds/checkpoint-25000.tar

# Visualize the pretrained JK2017 (Kinect) model
python main.py --dataset=jk2017-kinect --evaluate=pretrained_models/jk2017-kinect/checkpoint-25000.tar
```

## Additional Open Source Goodies

Our requirements for this work yielded several other standalone projects which we have also made public.

Our [soft DTW for PyTorch in CUDA](https://github.com/Maghoumi/pytorch-softdtw-cuda) project, is a fast implementation of the
[sDTW algorithm](https://arxiv.org/abs/1703.01541) on which DeepNAG's loss function relies. We additionally implemented 
second-order differentiable GRU units for PyTorch using TorchScript ([JitGRUs](https://github.com/Maghoumi/JitGRU)).  

## Support/Citing
If you find our work useful, please consider starring this repository and citing our work:

```
@misc{maghoumi2020deepnag,
      title={{DeepNAG: Deep Non-Adversarial Gesture Generation}}, 
      author={Mehran Maghoumi and Eugene M. Taranta II and Joseph J. LaViola Jr},
      year={2020},
      eprint={2011.09149},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Note that JK2017 dataset **IS NOT**
a part of this project, and has a [different license](https://raw.githubusercontent.com/ISUE/Jackknife/master/LICENSE).
