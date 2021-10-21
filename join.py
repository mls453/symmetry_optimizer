import cvfunc
import cv2
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Joins pictures side-by-side horizontally.')
    parser.add_argument('--input_folder', type=str, help='Input folder with pictures.')
    parser.add_argument('--output_folder', type=str, help='Output folder.')
    parser.add_argument('--keyword_left', type=str, help='Keyword for pictures to be placed on the left.')
    parser.add_argument('--keyword_right', type=str, help='Keyword for pictures to be placed on the right.')
    parser.add_argument('--bgr', type=int, help='Red value of background color (from zero to one).', default=1)
    parser.add_argument('--bgg', type=int, help='Green value of background color (from zero to one).', default=1)
    parser.add_argument('--bgb', type=int, help='Blue value of background color (from zero to one).', default=1)
    args = parser.parse_args()
    
    cvfunc.butterfly(
        folder=args.input_folder,
        destination=args.output_folder,
        left_keyword=args.keyword_left,
        right_keyword=args.keyword_right,
        bg_color=(args.bgr,args.bgg,args.bgb)
    )

if __name__=="__main__":
    main()