# Symmetry Optimizer

Project developed under the request of Michael Smith on Upwork.

This program will attempt to find the most symmetrical orientation for a set of pictures.


### Prerequisites

- Linux, MAC or Ubuntu subsystem for Windows (In case you're running it on Windows).
- Python3
- pip

## Installing

### Python requirements

With pip installed and in the project directory, run:

`pip -r install requirements.txt`

## Usage

There are three python scripts available. They are: `symmetry.py`, `scale.py` and `join.py`. Examples of usage are available in the `.sh` files. The file `cvfunc.py` is a module and it isn't intended to run on its own.

usage: symmetry.py [-h] [--input_folder INPUT_FOLDER] [--ncolor NCOLOR] [--output_folder OUTPUT_FOLDER] [--search_ratio SEARCH_RATIO] [--csv CSV]

Finds rigid transformation symmetries in pictures on x and y axes.

arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
                        Input folder with pictures.
  --ncolor NCOLOR       Number of colors to be used.
  --output_folder OUTPUT_FOLDER
                        Output folder.
  --search_ratio SEARCH_RATIO
                        Number from 0 to 1 for the search box.
  --csv CSV             Path to save csv metadata.

usage: scale.py [-h] [--input_folder INPUT_FOLDER] [--output_folder OUTPUT_FOLDER] [--scale_height SCALE_HEIGHT]

Resizes pictures.

arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
                        Input folder with pictures.
  --output_folder OUTPUT_FOLDER
                        Output folder.
  --scale_height SCALE_HEIGHT
                        Height of the final pictures.

usage: join.py [-h] [--input_folder INPUT_FOLDER] [--output_folder OUTPUT_FOLDER] [--keyword_left KEYWORD_LEFT] [--keyword_right KEYWORD_RIGHT] [--bgr BGR] [--bgg BGG] [--bgb BGB]

Joins pictures side-by-side horizontally.

arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
                        Input folder with pictures.
  --output_folder OUTPUT_FOLDER
                        Output folder.
  --keyword_left KEYWORD_LEFT
                        Keyword for pictures to be placed on the left.
  --keyword_right KEYWORD_RIGHT
                        Keyword for pictures to be placed on the right.
  --bgr BGR             Red value of background color (from zero to one).
  --bgg BGG             Green value of background color (from zero to one).
  --bgb BGB             Blue value of background color (from zero to one).

## Authors

* **Ícaro Lorran Lopes Costa** - (https://www.linkedin.com/in/icarolorran/)

