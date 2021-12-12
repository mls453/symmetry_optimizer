import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
import glob
from skimage import transform

def resize(img,scale_height):
    h,w,c = img.shape
    resized = cv2.resize(img,(int(w*scale_height/h),scale_height),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    return resized

def resize2(img,scale_height,s):
    # Source width and height in pixels
    src_w_px = img.shape[1]
    src_h_px = img.shape[0]

    # Target width and height in pixels
    res_w_px = int(w*scale_height/h)
    res_h_px = scale_height

    Affine_Mat_w = [s, 0, res_w_px/2.0 - s*src_w_px/2.0]
    Affine_Mat_h = [0, s, res_h_px/2.0 - s*src_h_px/2.0]

    M = np.c_[ Affine_Mat_w, Affine_Mat_h].T 
    return cv2.warpAffine(src_rgb, M, (res_w_px, res_h_px))

def resize3(img,scale_height):
    h,w,c = img.shape
    return transform.resize(
        img,
        (scale_height,int(w*scale_height/h)),
        mode="edge",
        anti_aliasing=False,
        anti_aliasing_sigma=None,
        preserve_range=True,
        order=0
    )

def main():
    
    parser = argparse.ArgumentParser(description='Resizes pictures.')
    parser.add_argument('--input_folder', type=str, help='Input folder with pictures.')
    parser.add_argument('--output_folder', type=str, help='Output folder.')
    parser.add_argument('--scale_height', type=int, help='Height of the final pictures.')
    args = parser.parse_args()
    
    assert os.path.exists(args.input_folder)
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    for file in tqdm(glob.glob(args.input_folder+"/*"),desc="Progress"):
        original = cv2.imread(file)
        #resized = resize(original,args.scale_height)
        #resized = resize2(original,args.scale_height,0.5)
        resized = resize3(original,args.scale_height)
        savestring = os.path.join(args.output_folder,f"{os.path.splitext(os.path.basename(file))[0]}.png")
        cv2.imwrite(savestring,resized)

if __name__=="__main__":
    main()