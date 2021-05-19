# EmoGIF


## Getting Started

### Requirements and Dependencies
* MUST HAVE OPENCV 4.5.0 (4.5.2 does not work)
* Python 3 (for Python 3.9, use opencv=4.5.0=py39_2)
* conda install -c conda-forge dlib
* conda install -c conda-forge scipy
* pip install opencv-contrib-python
* pip install h5py
* pip install trimesh==2.37.1
* pip install mls
* pip install scikit-image
* conda install -c kitware-danesfield-cf rtree 

In case something isn't working right, please see the requirements.txt file exported from my personal environment to see the versions I had while writing this code.

### Run Without GUI

`$python bringToLife.py <path-to-selfie-image> <emotion> <output-filename>`

For example: 
  
`$ python bringToLife.py input_images/Frida1.jpg peaceful Frida1-peaceful.mp4`

emotion must be one of:  excited, happy, whimsical, love, peaceful, nostalgic, gloomy, confused

output filename must be of type mp4
  
### Run With GUI

The GUI uses Zenipy. Please follow https://github.com/poulp/zenipy if you would like to use this.  Once set up, run:
  
`$ python bringToLife_gui.py`
  
Note that Zenipy does not handle typos or backspace in user input.
  
