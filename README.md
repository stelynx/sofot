# SOFOT

SOFOT (Stelynx Optical Flow-based Object Tracker) is a simple object
tracker implemented by Stelynx in Python.

## Table of Contents

- [Setup](#setup)
- [Datasets](#datasets)
- [Running](#running)
- [Algorithm](#algorithm)
- [License](#license)

## Setup

*SOFOT was developed and tested for Python 3.7.9 on MacBook Pro (16-inch,
2019), however it should work with any operating system and any Python
3.x (at least above 3.6). No GPU is required.*

Usage of Python virtual environment is recommended to mitigate version
conflicts of packages.

```
virtualenv venv
source venv/bin/activate
```

SOFOT depends on `numpy`, `scipy`, and `opencv-python`. You can install
them via `pip` or your preferred package manager. Mine is `pip`, so you
can just run

```
pip install -r requirements.txt
```

## Datasets

This projects was initially built for object detection on [MODD1 dataset](https://vision.fe.uni-lj.si/RESEARCH/modd/),
therefore the dataset you want to use should follow the same directory
structure for you to be able to use it out of the box. The directory
structure should be the following:

```
data
  |- dataset_1
  |    |- video_1
  |    |    |- images
  |    |    |    |- 00001.jpg
  |    |    |    |- 00002.jpg
  |    |    |    |- ...
  |    |    |    |- 0000N.jpg
  |    |    |- gt.mat
  |    |- video_2
  |    |    |- images
  |    |    |- gt.mat
  |    ...
  |    |- video_N
  |    |    |- images
  |    |    |- gt.mat
  |- dataset_2
  ...
sofot
```

where `gt.mat` is MATLAB file containing `largeobjects` and `smallojects`
values that are `N_FRAMES x 1` cell. Each value in the cell should be `4 x N_OBJECTS`
matrix, containing `[x1; y1; x2; y2]` columns for top left and lower
right corner of bounding box for each object.

For any other directory structure or annotations file, you must write a
function similar to `get_modd1_data()` in `util.py` and change `get_data(dataset)` to call your function.

### Using MODD1 for testing

You can use MODD1 for testing or for verification of installation.
There is a script `/data/download_datasets.sh` that downloads and prepares
MODD1 dataset for it to be used by SOFOT. From root of the project execute the following.

```
cd data
bash download_datasets.sh
cd ..
python sofot/main.py --dataset modd1
```

## Running

SOFOT's main script is `sofot/main.py`. Bare in mind that SOFOT is meant
to be run from project root directory!

SOFOT has the following command-line arguments, which can also be printed
to command line using `python sofot/main.py -h`.

<table>
  <tr>
    <th>Argument</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>-h</code>, <code>--help</code></td>
    <td>Prints help.</td>
  </tr>
  <tr>
    <td><code>--dataset</code></td>
    <td><b>Required.</b> Folder name in "data" folder to be used.</td>
  </tr>
  <tr>
    <td><code>--video</code></td>
    <td>Run only on this video.</td>
  </tr>
  <tr>
    <td><code>--benchmark</code></td>
    <td>Run in benchmark mode, rendering, saving, and debugging is disabled.</td>
  </tr>
  <tr>
    <td><code>--render</code></td>
    <td>Render video of frames with bounding boxes.</td>
  </tr>
  <tr>
    <td><code>--save-bbox</code></td>
    <td>Save bounding boxes in files for each frame. Format of bounding box is "x1 y1 x2 y2" (upper left and lower right corner).</td>
  </tr>
  <tr>
    <td><code>--debug</code></td>
    <td>Run in debug mode. Stops after each frame is presented and prints debugging information to stdout.</td>
  </tr>
</table>

To run benchmark on all videos in a dataset called `my_dataset`, you would run

```
python sofot/main.py --dataset my_dataset --benchmark
```

To generate a video of frames with bounding boxes and corresponding
annotations file, but only for video `video_14`, you would run

```
python sofot/main.py --dataset my_dataset --video video_14 --render --save-bbox
```

## Algorithm

Although the algorithm might seem trivial, it performs marvelously. The
algorithm behind SOFOT consists of two phases, **optical flow** for direct
object tracking and **enhanced IoU** for mapping of the detections.
In the following paragraphs the algorithm is described in greater detail.

### Initialisation

SOFOT starts with given bounding boxes for first frame. Using [Shi-Tomasi
corner detection algorithm](https://en.wikipedia.org/wiki/Corner_detection#The_Harris_&_Stephens_/_Shi–Tomasi_corner_detection_algorithms) it extracts good features for tracking inside
of each bounding box.

### Main loop

On each iteration, new set of feature points for each object is calculated
based on optical flow estimation. This is done using [Lucas-Kanade method](https://en.wikipedia.org/wiki/Lucas–Kanade_method).

After all new points have been calculated, new bounding boxes are
generated. Bounding boxes are generated from points simply by taking the
smallest rectangle enclosing all the points of the object. For each pair
in cartesian product of old bounding boxes and new bounding boxes, an
IoU (intersection over union) is calculated. Every old bounding box is
paired with a new bounding box so that their IoU is the highest.

When IoU matching is over and there are still some unmatched old bounding
boxes, that means that that old bounding box has IoU = 0 with all the
new bounding boxes, which means it either
exited the frame (in this case it is ignored from hereon), or it has
moved so much that it does not overlap with its new bounding box. If it
did not exit the frame, the closest bounding box from unmatched new ones
is found. For steady and slow moving environments like maritime this is
a completely justifiable and on-point assumption, because it is highly
unlikely that many of the objects being tracked would change their
position so drastically inside of one-frame time period that the change
in detected identity would occur.

On each iteration, calculated bounding boxes are also compared with
ground-truth ones. An error is reported on screen if there is a detected
bounding box that does not overlap with any bounding box in annotations.

## Benchmarks

Algorithm was benchmarked on [MODD1 dataset](https://vision.fe.uni-lj.si/RESEARCH/modd/)
and it produced extremely good results. We benchmarked frames-per-second processed and errors made and results are in the following table.

<table style="margin: 0 auto">
  <tr>
    <th style="text-align: center">Video</th>
    <th style="text-align: center">FPS</th>
    <th style="text-align: center">errors (at frame)</th>
  </tr>
  <tr>
    <td style="text-align: center">01</td>
    <td style="text-align: center">144</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">02</td>
    <td style="text-align: center">233</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">03</td>
    <td style="text-align: center">162</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">04</td>
    <td style="text-align: center">240</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">05</td>
    <td style="text-align: center">233</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">06</td>
    <td style="text-align: center">227</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">07</td>
    <td style="text-align: center">245</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">08</td>
    <td style="text-align: center">238</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">09</td>
    <td style="text-align: center">180</td>
    <td style="text-align: center">1 (108/108)</td>
  </tr>
  <tr>
    <td style="text-align: center">10</td>
    <td style="text-align: center">186</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">11</td>
    <td style="text-align: center">238</td>
    <td style="text-align: center">0</td>
  </tr>
  <tr>
    <td style="text-align: center">12</td>
    <td style="text-align: center">233</td>
    <td style="text-align: center">0</td>
  </tr>
</table>

## License

SOFOT was implemented as a final project for MSc course *Imaging technologies*
(Advanced Computer Vision) at University of Ljubljana, Faculty of
Electrical Engineering. It is licensed under [MIT License](LICENSE) and
therefore you are free to use this code. If you use it for research,
please cite as

```
@misc{Stelynx_SOFOT,
  author = {Stelynx},
  title = {SOFOT - Stelynx Optical Flow-based Object Tracker},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stelynx/sofot}}
}
```

and for any other project, mentions would be highly appreciated.
