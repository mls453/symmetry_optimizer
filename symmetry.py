import cvfunc
import cv2
from scipy.optimize import brute
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from time import time
import uuid
from scipy.stats import binom
from functools import partial

import logging 

logging.basicConfig(filename='symmetry.log', 
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s') 

#===============
#Symmetry scores
#===============

def symmetry(img):
    """
    Receives an image array X.
    The symmetry score is calculated with the mean squared error of a binary mask.
    An image with score zero indicates a complete symmetry whereas a score 1 indicates complete asymmetry.
    """
    if img.max()==255:
        img = (img/255).astype(np.uint8)
    img_xor = (img - cv2.flip(img,1))**2
    area = img.sum()
    chisq = img_xor.sum()/(area*area) if area!=0 else 0
    return chisq, area

def correl(img):
    if img.max()==255:
        img = (img/255).astype(np.uint8)
    area = img.sum()
    if area!=0:
        c = (img&cv2.flip(img,1)).ravel().sum()
        return (1-np.abs(c)/area), area
    else:
        return 0,0

def s(X):
    count = np.bitwise_and(X,X[:,::-1]).sum()
    return count
    
def symmetry_stochastic(img,color_dict):
    layers, _ = cvfunc.img_to_layer_mask(img,color_dict)
    s_layer = list(map(s,layers))
    n = np.count_nonzero(s_layer)
    S = sum(s_layer) #Computed value
    A = np.array(layers).sum(axis=0)
    M = np.bitwise_and(A,A[:,::-1]).sum()
    M = M/A.sum() #Relative pair count
    P = binom(M,1/n)
    E = M/n #Expectation value
    std = P.std()
    #return (2/(1+np.exp((S-E)/A.sum())/std/np.sqrt(2)))-1
    return 2*P.cdf(S/A.sum()) - 1

#===================================================


def find_optimal_angle(img,sample,invert,box,l,ratio=0.05,theta=0,opt=symmetry,auto_crop=True):
    """
    Tries to find the least symmetry score of an image by rotating and translating its contents.
    There is one symmetry score calculated per color and the total is summed up and fed to an optimizer.
    Theta is the rotation angle in degrees.
    """
    _id = str(uuid.uuid1())[:8]
    
    angle = theta
    
    layers = [cvfunc.color_threshold(img,[color[:3]],[color[3:]],invert) for color in sample]
    coords = None
    
    s = sum(map(lambda x: x.sum(),layers))
    if s==0:
        logging.info(f"{_id}\tfind_optimal_angle\tzero sum picture")
        return [np.nan,np.nan],False
    
    if len(layers)==1 and auto_crop:
        coords = cvfunc.get_bounding_box(layers[0])
        layers[0] = cvfunc.auto_crop(layers[0])
        h,w = layers[0].shape
        box_center_y = int(h/2)
        box_center_x = int(w/2)
        w = ratio*w
    else:
        box_center_x = int((box[0]+box[2])/2)
        box_center_y = int((box[1]+box[3])/2)
        h = (box[3]-box[1])*ratio
        w = (box[2]-box[0])*ratio

    if h==0 and w==0:
        logging.info(f"{_id}\tfind_optimal_angle\tzero shaped picture")
        return [0,0],coords
    else:
        x_low, x_high = box_center_x-int(w/2), box_center_x+int(w/2)

        def f(par):
            x = int(par[0])
            S = np.array(list(map(lambda im: opt(cvfunc.rotate_image(im,angle,(x,box_center_y),l)),layers)))
            weighted_sum = (S[:,0]*S[:,1]).sum()
            area = S[:,1].sum()
            score = weighted_sum/area if area != 0 else 0
            return score

        #res = brute(f, (search_box_x,))
        logging.info(f"{_id}\tfind_optimal_angle\toptimizing over range\t{w}")
        t = time()
        brute = [(x,f([x])) for x in np.arange(x_low,x_high+1)]
        res = min(brute,key=lambda z: z[1])
        x = res[0]
        #fmin = f(res)
        fmin = res[1]
        res = [x,fmin]
        logging.info(f"{_id}\tfind_optimal_angle\tRUNTIME\t{time()-t}")
        return res,coords

def savefig(img,n,file,output_folder):
    cv2.imwrite(os.path.join(output_folder,f"{os.path.splitext(os.path.basename(file))[0]}_{n}_.png"), img)

def process_one(img,search_ratio,color_dict=None,ncolor=None,angle=0,auto_crop=True):
    t = time()
    """
    img: opencv2 object
    
    If `color_dict` is None, then uses value specified by `ncolor` to find color space by K-Means.
    Otherwise, it uses the values established by `color_dict` itself. The color called "bg" will be considered the backgound color.
    """
    data = []
    generated_figures = []
    
    if color_dict!=None:
        white_ix = "bg"
        colors = color_dict
        im_reduced = img
        ncolor = len(color_dict) - 1
        values = np.array(list(colors.values()))
        if values.max()>1 or issubclass(values.dtype.type,np.integer):
            values = values/255.
            colors = dict(zip(colors.keys(),values.tolist()))
    else:
        im_reduced = cvfunc.reduce_colors(img,n=ncolor)
        colors = cvfunc.get_colors(im_reduced)[0] 
        white_ix = ((1-colors)**2).sum(axis=1).argmin()
        colors = dict(enumerate([list(x)+list(x) for x in colors]))
    w = tuple(colors[white_ix])
    im_white_out = cvfunc.color_threshold(im_reduced,[w[:3]],[w[3:]],invert=True)
    box = cvfunc.get_bounding_box(im_white_out)
    l = 2*int(np.ceil(np.sqrt((box[2]-box[0])**2 + (box[3]-box[1])**2)))
    
    full_colors = colors.copy()
    del full_colors[white_ix]
    full_colors = [tuple(x) for x in full_colors.values()]
    
    for n,color in colors.items():
        color = tuple(color)
        
        try:
            assert len(color)==6
        except:
            print(n,color)
            print(colors)
            1/0
        
        label = n
        im_color=0
        im_ideal=0
        logging.info(f"color\t{n}")
        if n!=white_ix:
            """
            Runs for a single color.
            """            
            params = find_optimal_angle(im_reduced,sample=[color],invert=False,box=box,l=l,ratio=search_ratio,theta=angle,opt=symmetry,auto_crop=auto_crop)
            xmse,fmse = np.array(params[0])
            
            if params[1]==False:
                logging.info(f"process_one\tskipping {n} because coords is {params[1]}")
                continue
            
            params = find_optimal_angle(im_reduced,sample=[color],invert=False,box=box,l=l,ratio=search_ratio,theta=angle,opt=correl,auto_crop=auto_crop)
            xcor,fcor = np.array(params[0])
            coords = params[1]
            
            
            if coords:
                x1,y1,x2,y2 = coords
                im_reduced_ = im_reduced[x1:x2,y1:y2]
            else:
                im_reduced_ = im_reduced[:,:]
            try:
                im_ideal = cvfunc.color_threshold(im_reduced_,[color[:3]],[color[3:]],invert=False)
                im_ideal = cvfunc.rotate_image(im_ideal,angle,(xcor,int(np.floor(im_ideal.shape[0]/2))),l)
                im_color = cvfunc.rotate_image(im_reduced_,angle,(xcor,int(np.floor(im_ideal.shape[0]/2))),l)
            except:
                pass
        else:
            """
            Runs when the color is 'white'.
            This part is responsible to find symmetry for all the colors combined.
            """
            label = "composition"
            
            params = find_optimal_angle(im_reduced,sample=full_colors,invert=False,box=box,l=l,ratio=search_ratio,theta=angle,opt=symmetry,auto_crop=auto_crop)
            xmse,fmse = np.array(params[0])
            
            if params[1]==False:
                logging.info(f"process_one\tskipping {n} because coords is {params[1]}")
                continue
            
            params = find_optimal_angle(im_reduced,sample=full_colors,invert=False,box=box,l=l,ratio=search_ratio,theta=angle,opt=correl,auto_crop=auto_crop)
            xcor,fcor = np.array(params[0])
            
            coords = params[1]
            if coords:
                x1,y1,x2,y2 = coords
                im_white_out_ = im_white_out[x1:x2,y1:y2]
                im_reduced_ = im_reduced[x1:x2,y1:y2]
            else:
                im_white_out_ = im_white_out[:,:]
                im_reduced_ = im_reduced[:,:]
            try:
                im_ideal = cvfunc.rotate_image(im_white_out_,angle,(xcor,int(np.floor(im_white_out.shape[0]/2))),l)
                im_color = cvfunc.rotate_image(im_reduced_,angle,(xcor,int(np.floor(im_white_out.shape[0]/2))),l)
            except:
                pass
            #TAB BACK
        try:
            im_color = im_color*255
            im_ideal = cv2.bitwise_and(im_color.astype(np.uint8),im_color.astype(np.uint8),mask = im_ideal.astype(np.uint8))
            generated_figures.append({"fig":im_ideal,"id":label})
        except:
            pass
        data.append({"angle":angle,"x_MSE":xmse,"score_MSE":fmse,"x_COR":xcor,"score_COR":fcor,"id":label,"R":color[0],"G":color[1],"B":color[2]})
    print("process_one",time()-t)
    return data, generated_figures

def f(x,angle,search_ratio,color_dict):
    img,filename = x
    data,generated_figures = process_one(img,search_ratio,color_dict=color_dict,angle=0)
    for d in data:
        d.update({"filename": filename})
    return data

def main():
    """
    The program starts here.
    """
    
    parser = argparse.ArgumentParser(description='Finds rigid transformation symmetries in pictures on x and y axes.')
    parser.add_argument('--input_folder', type=str, help='Input folder with pictures.')
    parser.add_argument('--ncolor', type=int, help='Number of colors to be used.')
    parser.add_argument('--output_folder', type=str, help='Output folder.')
    parser.add_argument('--search_ratio', type=float, help='Number from 0 to 1 for the search box.')
    parser.add_argument('--csv', type=str, help='Path to save csv metadata.')
    args = parser.parse_args()
    
    assert os.path.exists(args.input_folder)
    
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    data = []
    
    files = next(os.walk(args.input_folder))[-1]
    for file in tqdm(files, desc="Progress"):
        data+=process(os.path.join(args.input_folder,file),args.ncolor,args.output_folder,args.search_ratio)
    
    pd.DataFrame(data).to_csv(args.csv,index=False)

if __name__=="__main__":
    main()

