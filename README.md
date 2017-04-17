# School Software Project
## Summer Semester

Neural network able to predict frames of an video.

About
==========
Whole project is divided into three parts - three main networks. Each part contains several modifications of the base network for the given part. The whole project is detailed described in `docs/rp_specifikace.pdf_` (in Czech language).


Naming convention of these parts is as follows:

  *`nn_part_one`
   *First part of the project. Contains base network in the folder `_base_` and three modifications in folders `modif_1`, `modif_2` and `modif_3` respectively.

How to run
==========
`cd` to the specific part and then to either base model or one of modification models. Videos for every network must be generated - run `python3 generate_videos_<base|modif_1|modif_2|modif_3>` base on the model you want to run in your terminal. 

After videos are generated you can train and test the network by typing `python3 <base|modif_1|modif_2|modif_3>.py` in your terminal.

Every model saves its logs into the `logs` folder. These logs can be then visualized using Tensorboard by running command `tensorboard --logdir="path/to/logs"`

Tech stack
==========
Written in Python3
[Tensorflow](tensorflow.org) for the creation of a neural network.
[NumPy](http://www.numpy.org/) for the data handling
[OpenCV](http://opencv.org/) for the video data handling

Results
=======
Here will be a table of accuracy for all models.
