import cv2
import numpy as np
import os
import glob
import tqdm
from time import time

"""
This module constains all the necessary functions to operate on the image files.
"""

def numpy_to_XYZ(x):
    if issubclass(x.dtype.type,np.integer):
        return x
    else:
        return (x*255).astype(np.uint8)

def tuple_to_XYZ(x):
    if len(x)==0:
        return x
    elif isinstance(x[0],int):
        return x
    else:
        return tuple([int(255*y) for y in x])

def get_colors(img):
    """
    Gets a list of all the colors used in the image.
    """
    all_rgb_codes = img.reshape(-1,img.shape[-1])
    colors = np.unique(all_rgb_codes, axis=0, return_counts=True)
    return colors

def reduce_colors(img,n=16):
    """
    Uses K-Means clustering to reduce color space.
    """
    img_data = img/255.0
    img_data = img_data.reshape(-1,img_data.shape[-1])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    img_data = img_data.astype(np.float32)
    compactness, labels, centers = cv2.kmeans(img_data,n, None, criteria,10, flags)
    new_colors = centers[labels].reshape((-1, centers[labels].shape[-1]))
    return new_colors.reshape(img.shape)

def color_threshold(img, sample_low=[[1,1,1]], sample_high=[[1,1,1]], invert=False):
    """
    Creates a mask for the image by the colors provided in a range from `sample_low` to `sample_high`.
    """
    assert len(sample_low)==len(sample_high)
    
    img = numpy_to_XYZ(img)
    sample_low = list(map(tuple_to_XYZ,sample_low))
    sample_high = list(map(tuple_to_XYZ,sample_high))
    
    composition = np.zeros(img.shape[:2])
    for l,h in zip(sample_low,sample_high):
        try:
            composition += cv2.inRange(img,tuple(l),tuple(h))
        except:
            pass
    if invert:
        composition = 255 - composition
    return composition

def color_filter(img,palette):
    """
    Only allows colors specified in the BGR color ranges in the palette dictionary.
    """
    img = numpy_to_XYZ(img)
    canvas = np.zeros(img.shape).astype(np.uint8)
    
    for name,color_tuple in palette.items():
        sample_low = color_tuple[:3]
        sample_high = color_tuple[3:]
        
        mask = color_threshold(img,[sample_low],[sample_high],invert=False)
        mask = np.stack([mask,mask,mask],axis=-1)
        mask /= 255.0
        mask = mask.astype(np.uint8)
        
        canvas += np.array(sample_low).astype(np.uint8)*mask
    return canvas

def get_center(mask):
    """
    Gets centroid of a mask.
    """
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)

def get_diam(img):
    """
    Gets bounding box diagonal length.
    """
    x,y,w,h = cv2.boundingRect(img.astype(np.uint8))
    return np.sqrt(w*w+h*h)

def get_bound_area(img):
    """
    Gets bounding box area.
    """
    x,y,w,h = cv2.boundingRect(img.astype(np.uint8))
    return w*h

def get_bounding_box(img):
    """
    Gets bounding box coordinates.
    """
    x,y,w,h = cv2.boundingRect(img.astype(np.uint8))
    return x,y,x+w,y+h 

def auto_crop(img):
    x1,y1,x2,y2 = get_bounding_box(img)
    return img[y1:y2,x1:x2]

def rotate_image(image, angle, pivot, l):
    """
    Rotates the image by an angle provided by `angle` (in degrees) relative to `pivot` and centers it also using `pivot`.
    """
    #l = int(np.ceil(get_diam(image)))
    #rot_mat = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    try:
        T = np.array([
            [1,0,l/2-pivot[0]],
            [0,1,l/2-pivot[1]]
        ])
        result = cv2.warpAffine(image, T, (l,l))
        if angle==0:
            result = image
        elif angle==90:
            result = image.T
        else:
            rot_mat = cv2.getRotationMatrix2D((l/2,l/2), angle, 1.0)
            result = cv2.warpAffine(result, rot_mat, (l,l))
        return result
    except:
        image

def butterfly(folder, destination, left_keyword, right_keyword, bg_color=(1,1,1)):
    """
    Used to join pictures side-by-side. 
    """
    assert os.path.exists(folder)
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    left = sorted(glob.glob(os.path.join(folder,f"*{left_keyword}*")))
    right = sorted(glob.glob(os.path.join(folder,f"*{right_keyword}*")))
    
    assert len(left)==len(right)
    
    for l_name,r_name in tqdm.tqdm(zip(left,right),desc="Progress",total=len(left)):
        l_original, r_original = cv2.imread(l_name)/255, cv2.imread(r_name)/255
        l,r = color_threshold(l_original,sample_low=[bg_color],sample_high=[bg_color],invert=True), color_threshold(r_original,sample_low=[bg_color],sample_high=[bg_color],invert=True)
        bboxl = cv2.boundingRect(l.astype(np.uint8))
        bboxr = cv2.boundingRect(r.astype(np.uint8))
        
        l_original = (l_original*255).astype(np.uint8)
        r_original = (r_original*255).astype(np.uint8)
        
        l = cv2.bitwise_and(l_original,l_original,mask=l.astype(np.uint8))
        r = cv2.bitwise_and(r_original,r_original,mask=r.astype(np.uint8))
        
        invert = False
        
        hscale = bboxr[3]/bboxl[3]
        if hscale<1:
            hscale = 1/hscale
            a = r
            r = l.copy()
            l = a.copy()
            invert = True
            
            b = bboxr
            bboxr = bboxl
            bboxl = b
        
        l = l[bboxl[1]:bboxl[1]+bboxl[3],bboxl[0]:bboxl[0]+bboxl[2]]
        r = r[bboxr[1]:bboxr[1]+bboxr[3],bboxr[0]:bboxr[0]+bboxr[2]]
        
        l = cv2.resize(l,(int(hscale*bboxl[2]),bboxr[3]))
        
        canvas = np.concatenate((l, r) if not invert else (r, l), axis=1)
        
        savestring = os.path.join(destination,f"{os.path.splitext(os.path.basename(l_name))[0]}_{os.path.splitext(os.path.basename(r_name))[0]}.png")
        cv2.imwrite(savestring,canvas)

def palette_guess(img,palettes):
    img_color_set = set(map(str,get_colors(img)[0].tolist()))
    guesses = []
    for k,palette in palettes.items():
        p = set(map(str,np.array(list(palette.values()))[:,:3].tolist()))
        intersection = p&img_color_set
        if len(intersection) > 1:
            guesses.append((k,len(intersection)))
    return max(guesses,key=lambda x: x[1])[0]