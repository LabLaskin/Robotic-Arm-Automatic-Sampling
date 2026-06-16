import cv2
import numpy as np
# from PIL import Image
from skimage import io, transform
# from skimage.measure import block_reduce
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import os


def hyab(img1, img2):
    HyAB = np.abs(img1[:,:,0]-img2[:,:,0]) + np.sqrt( (img1[:,:,1]-img2[:,:,1])**2 + (img1[:,:,2]-img2[:,:,2])**2 )
    return HyAB

def get_hyab_gradient_img(img):
    h,w,c = img.shape
    z1 = [0, h-2, 0, w-2]
    z2 = [0, h-2, 1, w-1]
    z3 = [0, h-2, 2, w  ]
    z4 = [1, h-1, 0, w-2]
    # z5 = [1, h-1, 1, w-1]
    z6 = [1, h-1, 2, w  ]
    z7 = [2, h  , 0, w-2]
    z8 = [2, h  , 1, w-1]
    z9 = [2, h  , 2, w  ]
    
    # gx = hyab(z1,z7) + 2*hyab(z2,z8) + hyab(z3,z9)
    gx = hyab(img[z1[0]:z1[1], z1[2]:z1[3]], img[z7[0]:z7[1], z7[2]:z7[3]]) + \
         (2 * hyab(img[z2[0]:z2[1], z2[2]:z2[3]], img[z8[0]:z8[1], z8[2]:z8[3]])) + \
         hyab(img[z3[0]:z3[1], z3[2]:z3[3]], img[z9[0]:z9[1], z9[2]:z9[3]])
    
    # gy = hyab(z1,z3) + 2*hyab(z4,z6) + hyab(z7,z9)
    gy = hyab(img[z1[0]:z1[1], z1[2]:z1[3]], img[z3[0]:z3[1], z3[2]:z3[3]]) + \
         (2 * hyab(img[z4[0]:z4[1], z4[2]:z4[3]], img[z6[0]:z6[1], z6[2]:z6[3]])) + \
         hyab(img[z7[0]:z7[1], z7[2]:z7[3]], img[z9[0]:z9[1], z9[2]:z9[3]])
    # hyab(lr_img[:lr_img.shape[0]-2, :lr_img.shape[1]-2], lr_img[2:lr_img.shape[0], :lr_img.shape[1]-2])

    gradient_img = np.abs(gx) + np.abs(gy)
    gradient_img = gradient_img/gradient_img.max()

    output = np.zeros((h,w), dtype = gradient_img.dtype)
    output[1:h-1, 1:w-1] = gradient_img
    return output

def get_sobel_gradient_img(img):
    gx = sobel(img, axis = 0,  mode = 'nearest')
    gy = sobel(img, axis = 1,  mode = 'nearest')
    if len(img.shape) == 3:
        gradient_img = np.sum(np.abs(gx), axis = 2) + np.sum(np.abs(gy), axis = 2)
    elif len(img.shape) == 2:
        gradient_img = np.abs(gx) + np.abs(gy)

    gradient_img = gradient_img/gradient_img.max()
    return gradient_img

def show(img):
    plt.imshow(img, interpolation = 'none')
    plt.show()
    plt.close()

def create_circular_mask(h, w, radius = None):

    # if center is None: # use the middle of the image
    center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask.astype(np.uint8)
mask = create_circular_mask(5, 5,)

def get_contours(gradient_img, kernel_size = 5, display = True, threshold_for_tissue = None):
    # get circular kernel for the closing morphological transform
    kernel = create_circular_mask(kernel_size,kernel_size)

    # spike filter
    clip_thre = np.quantile(gradient_img, .95)
    img = np.clip(gradient_img/clip_thre, 0,1)
    # increase contrast of image
    img = np.clip(((img-.99)*3)+.99, 0, 1)

    # Threshold high contrast image
    if threshold_for_tissue == None:
        thre, x = cv2.threshold((img*255).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        if type(threshold_for_tissue) == int:
            threshold_for_tissue = float(threshold_for_tissue/255)
        thre = threshold_for_tissue
        x = np.where(img > thre, 255, 0).astype(np.uint8)  
        x = np.repeat(x[:,:,None], repeats = 1, axis = 2)    

    # close small holes in contours
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)
    
    # find contours
    contours, _ = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if display:
        show_contours(contours, gradient_img)
    return contours

def show_contours(contours, img):
    mask = np.zeros(img.shape, np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, (255,255,255), -1)
    show(mask)

def get_tissue_contour_with_slide_in_image(contours):
    # Find size of each contour
    sizelist = [cv2.contourArea(contour) for contour in contours]
    # get the size of the contour from the glass slide (or holder if zoomed out even further)
    slide_contour = np.max(sizelist)

    # Assume tissue will be less than half the area of the slide but more than 1/40 the area.
    # Selects the largest contour fulfilling this criterion
    biggest_contour = None
    for i in np.argsort(-np.array(sizelist)):
        if (sizelist[i] > slide_contour/50) and (sizelist[i] < slide_contour/3):            
            biggest_contour = i
            break
    
    if biggest_contour == None:
        raise RuntimeError(
'''Failed to find the sample contour.
Please ensure that the sample takes up enough of the image area and
that there is sufficient contrast between the background and sample'''
)
    
    return biggest_contour


def show_tissue_mask(masks, img):
    if type(masks) == type(np.empty(0)):
        if masks.ndim == 2:
            masks = masks[np.newaxis]

    elif type(masks) not in [list, tuple]:
        masks = [masks]

    for mask in masks:
        if img.max():
            img = (img/img.max())
        mask = np.where(mask[:,:,None], img, 0)
        show(mask)


def compute_l2_distance(x, centroid):
    # Compute the difference, following by raising to power 2 and summing
    dist = ((x - centroid) ** 2).sum(axis = x.ndim - 1)
    return dist

def get_closest_centroid(x, centroids):
    
    # Loop over each centroid and compute the distance from data point.
    dist = compute_l2_distance(x, centroids)

    # Get the index of the centroid with the smallest distance to the data point 
    closest_centroid_index =  np.argmin(dist, axis = 1)
    
    return closest_centroid_index

def compute_sse(data, centroids, assigned_centroids):
    # Initialise SSE 
    sse = 0

    # Compute SSE
    sse = compute_l2_distance(data, centroids[assigned_centroids]).sum() / len(data)
    
    return sse

class KMeans:
    def __init__(self, n_clusters = 8, n_iter = 10, max_iter = 25, tol = 1e-4, random_seed = np.random.randint(1024**2)):
        # # Size of dataset to be generated. The final size is 4 * data_size
        # data_size = 1000
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.tol = tol
        self.n_iter = n_iter
        self.data = None
        self.centroids = None
        self.num_centroid_dims = None
        self.assigned_centroids = None
        self.sse = None

        np.random.seed(random_seed)

    
    def fit(self, data):

        self.data = np.array(data)
        
        # Number of dimensions in centroid
        self.num_centroid_dims = self.data.shape[1]

        # Shuffle the data
        rng = np.random.default_rng()
        
        final_centroids = []
        final_assigned_centroids = []
        final_sse = []

        for n in range(self.n_iter):
            self.shuffle_mask = rng.permutation(self.data.shape[0], axis = 0)
            self.unshuffle_mask = np.argsort(self.shuffle_mask)
            self.data_shuffled = self.data[self.shuffle_mask]

            # Initialise centroids
            self.initialize_centroids()
            # self.centroids = self.data_shuffled[np.random.randint(0, self.data_shuffled.shape[0], size = self.n_clusters)]


            # List to store SSE for each iteration 
            sse_list = []

            # Main Loop
            for i in range(self.max_iter):

                # Get closest centroids to each data point
                self.assigned_centroids = get_closest_centroid(self.data_shuffled[:, None, :], self.centroids[None,:, :])    
                
                # Compute new centroids
                # print(self.centroids.shape[1])
                for c in range(self.centroids.shape[0]):
                    # Get data points belonging to each cluster 
                    cluster_members = self.data_shuffled[self.assigned_centroids == c]
                    # Compute the mean of the clusters
                    if cluster_members.shape[0]:
                        cluster_members_mean = cluster_members.mean(axis = 0)
                    
                    # print(self.centroids.shape)
                    # Update the centroids
                    self.centroids[c] = cluster_members_mean
                
                # Compute SSE
                # print('a', self.data_shuffled, '\nb', self.centroids, '\nc', self.assigned_centroids)
                sse = compute_sse(self.data_shuffled, self.centroids, self.assigned_centroids)
                sse_list.append(sse)

                if (len(sse_list) > 3):
                    # if the sse is greater than the previous two calculated sse plus tolerance, then end
                    if (sse_list[-1] >= sse_list[-2]*(1+self.tol)) and (sse_list[-1] >= sse_list[-3]*(1+self.tol)):
                        break
            
            # get assignments for unshuffled data for this iteration
            self.assigned_centroids = self.assigned_centroids[self.unshuffle_mask]
            self.sse = sse_list[-1]

            final_centroids.append(self.centroids)
            final_assigned_centroids.append(self.assigned_centroids)
            final_sse.append(self.sse)

        idx_of_best_clusters = np.argmin(final_sse)
        self.centroids = final_centroids[idx_of_best_clusters]
        self.assigned_centroids = final_assigned_centroids[idx_of_best_clusters]
        self.sse = final_sse[idx_of_best_clusters]

        return self.assigned_centroids

    def initialize_centroids(self):
        self.centroids = np.empty((self.n_clusters, self.data_shuffled.shape[1]), dtype = np.float64)
        self.centroids[0] = self.data_shuffled[np.random.randint(0, self.data_shuffled.shape[0], size = 1)]

        for i in range(self.n_clusters-1):
            # calculate distance from all centroids
            dists = np.sum(np.square([np.sum(np.square(self.data_shuffled-self.centroids[j]), axis = 1) for j in range(i+1)]), axis = 0)
            # normalize distances
            dists /= np.sum(dists)
            # pick next centroid based on dists
            new_centroid_idx, = np.random.choice(range(self.data_shuffled.shape[0]), size=1, p=dists)

            self.centroids[i+1] = self.data_shuffled[new_centroid_idx]

def intersect(line1, line2, segment = True):
    p1, p2 = line1[0:2], line1[2:4] 
    p3, p4 = line2[0:2], line2[2:4] 

    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return np.array([np.nan, np.nan])
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if segment:
        if ua < 0 or ua > 1: # out of range
            return np.array([np.nan, np.nan])
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if segment:
        if ub < 0 or ub > 1: # out of range
            return np.array([np.nan, np.nan])
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return np.array([x, y])

def get_elbow_point(sse_list, clust_range):
    # use log to reduce the effect of the large error drop going from 1 to 2 clusters
    sse_list = np.log(sse_list)

    # get the slope of the line connecting the first and last points of the sse vs num_clusters plot
    slope = (sse_list[-1] - sse_list[0])/(clust_range[-1] - clust_range[0])
    
    # get the y-intersept of the line
    y_int = sse_list[0] - (slope * clust_range[0])

    # get the slope normal to the slope 
    slopePerp = -1/(slope)

    # collecter for distances from line
    distances = []
    # for each point, calculate the distance to the line
    for i, sse in enumerate(sse_list[1:-1]):
        
        # Define second point of line segment at the first value of sse
        x = ((sse-sse_list[0])/slopePerp) + clust_range[i+1]
        
        # find the intersection of the perpendicular lines
        line1 = [clust_range[0], sse_list[0], clust_range[-1], sse_list[-1]]
        line2 = [clust_range[i+1], sse, x, sse_list[0]]
        intersection = intersect(line1, line2, segment = False)

        # distance between sse point and the intersection
        distance = np.sqrt(
            np.sum(
                ((intersection) - (np.array([clust_range[i+1], sse])))**2
                )
            )
        
        # make the distance negative if the point is above the line between the first and last points
        if intersection[1] > ((slope * x) + y_int):
            distance = -distance

        distances.append(distance)
    # distances.append(0)
        
    # find the point with the maximum distance, which is the elbow point
    idx = np.argmax(distances)

    return idx+1

# resize masks to full size images
def generate_mask(contours, contour_idx, img, full_img):
    mask = np.zeros(img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, contour_idx, (255,255,255), -1)
    mask = transform.resize(mask, full_img[:,:,0].shape, order = 0)
    return mask

# def get_plus_contours(reference_images, full_img, thresholds):
#     # ref_im = reference_image
#     # lower_size_thre, upper_size_thre = size_threshold

#     # collecter for selected contours
#     # plus_contours = []
#     centerpoints = []
#     img = np.mean(full_img/full_img.max(), axis = 2)
    
#     # remove spikes in signal
#     spike_cutoff = np.quantile(img, .95)
#     img = np.clip(img, 0, spike_cutoff)*spike_cutoff
#     img = cv2.blur(img,(5,5),0)
#     img = (((img/img.max())-.4)*2)+.4
#     img = np.clip(img, 0, 1)
#     # show(img)
#     # img = (img*255).astype(np.uint8)
#     # img = img/img.max()

#     if len(img.shape) == 3:
#         img = img.max(axis = 2)
#     # run for each provided threshold
#     for thre in thresholds:
#         # threshold image
#         img_thre = np.where(img>thre, 255, 0).astype(np.uint8)
#         # show(img_thre)

#         # Group contiguous sets of pixels
#         labelled_grps = scipy.ndimage.label(img_thre, structure=np.ones((3,3)), output=None)[0]
#         _, count = np.unique(labelled_grps, return_counts = True)

#         # Draw an image that only contains the groups meeting the size and shape thresholds
#         redraw = np.zeros(img_thre.shape)
#         kept = []
#         for idx, i in enumerate(count):
#             # size thresholds determined empirically
#             if (i > img_thre.size//8000) and (i < img_thre.size//500):
#                 # ensure the shape is not oblong
#                 where_arr = np.array(np.where(labelled_grps == idx))
#                 aspect = where_arr.max(axis = 1)-where_arr.min(axis = 1)
#                 if aspect.max()/aspect.min() < 1.2:
#                     kept.append(idx)
#                     redraw[np.where(labelled_grps == idx)] = idx

#         hu_diffs = []
#         ref_im_moments = []

#         for ref_im in reference_images:
#             ref_im = np.load(ref_im)

#             cont = cv2.findContours(ref_im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             output = np.zeros(shape.shape, dtype=np.uint8)
#             cv2.drawContours(output, cont[0], -1, 255)
#             shape_moments = cv2.HuMoments(cv2.moments(output)).flatten()
#             Hi = -np.sign(shape_moments)*np.log10(np.abs(shape_moments))
#             ref_im_moments.append(Hi)
        
#         for i in kept:
#             ref_hu_diffs = []
#             pixels = np.array(np.where(redraw == i))
#             x_max, y_max = np.max(pixels, axis = 1)
#             x_min, y_min = np.min(pixels, axis = 1)
#             shape = redraw[x_min:x_max, y_min:y_max]

#             cont = cv2.findContours(shape.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             output = np.zeros(shape.shape, dtype=np.uint8)
#             cv2.drawContours(output, cont[0], -1, 255)
#             shape_moments = cv2.HuMoments(cv2.moments(output)).flatten()
#             Hi = -np.sign(shape_moments)*np.log10(np.abs(shape_moments))

#             for ref_Hi in ref_im_moments:
#                 ref_hu_diffs.append(L2_dif(Hi, ref_Hi))

#             hu_diffs.append(np.min(ref_hu_diffs))

#         sorted_hu_diffs = np.argsort(hu_diffs)

#         if sorted_hu_diffs.size:
#             mask = np.where(redraw == kept[sorted_hu_diffs[-1], 255, 0])
#             c1 = scipy.ndimage.center_of_mass(mask, labels=None, index=None)
#             centerpoints.append(c1)
#         if sorted_hu_diffs.size>1:
#             mask = np.where(redraw == kept[sorted_hu_diffs[-2], 255, 0])
#             c2 = scipy.ndimage.center_of_mass(mask, labels=None, index=None)
#             centerpoints.append(c2)





        
        
#         # #get contours of image
#         # contours, _ = cv2.findContours(img_thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         # # Determine similarity with hu moments and collect area of shapes
#         # hu_differences = []
#         # shape_sizes = [cv2.contourArea(contour) for contour in contours]
#         # meets_size_thre = np.where((np.array(shape_sizes) > lower_size_thre)&(np.array(shape_sizes) < upper_size_thre), True, False)

#         # for i, contour in enumerate(contours):
#         #     if meets_size_thre[i]:
#         #         minx, miny = contour.min(axis = 0).flatten()
#         #         maxx, maxy = contour.max(axis = 0).flatten()
#         #         shape = img_thre[miny+1:maxy,minx+1:maxx].astype(np.uint8)
#         #         hu_differences.append(cv2.matchShapes(ref_im,shape,cv2.CONTOURS_MATCH_I2,0.0))
#         #         # print(shape_sizes[i], size_thre)
#         #         # show_contours([contour], full_img)

#         #     else:
#         #         hu_differences.append(1000)

#         # # initialize collecter for indices of marker contours and the minimum size threshold for shapes to be considered
#         # plus_markers = []

#         # # sort by hu_differences
#         # sort_mask = np.argsort(np.array(hu_differences))
#         # # get the hu moment differences for shapes that are larger than 1/5000th the original image to remove tiny speckles
#         # meets_size_thre = np.where((np.array(shape_sizes) > lower_size_thre)&(np.array(shape_sizes) < upper_size_thre))[0]

#         # # find the 2 smallest differences that meet the size threshold
#         # for i in sort_mask:
#         #     if i in meets_size_thre:
#         #         plus_markers.append(i)
#         #         if len(plus_markers)>=2:
#         #             break
        
#         # for idx in plus_markers: 
#         #     plus_contours.append(contours[idx])
#         #     # show_contours(plus_contours, full_img)

#     return centerpoints

def sharpen_image_for_marker_detection(full_img, slide_brightness_estimate=.5, do_nothing = False):
    # Normalize image to [0-1] and make it grayscale
    img = np.mean(full_img/full_img.max(), axis = 2)
    
    if not do_nothing:
        # remove spikes in signal
        spike_cutoff = np.quantile(img, .95)
        img = np.clip(img, 0, spike_cutoff)*spike_cutoff

        # Blur to remove abberations
        img = cv2.blur(img,(5,5),0)

        # increase contrast
        img = (((img/img.max())-slide_brightness_estimate)*2)+slide_brightness_estimate
        img = np.clip(img, 0, 1)

    # ensure the image is grayscale
    if len(img.shape) == 3:
        img = img.max(axis = 2)

    return img

def get_reference_image_hu_moments(reference_images):
    ref_im_moments = []

    for ref_im in reference_images:
        ref_im = np.load(ref_im)
        ref_im = np.pad(ref_im, 1, mode='constant')

        # Get contours from ref_im
        cont = cv2.findContours(ref_im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours onto a blank image
        output = np.zeros(ref_im.shape[:2], dtype=np.uint8)
        cv2.drawContours(output, cont[0], -1, 255)

        # Find and save the log normalized Hu moments 
        shape_moments = cv2.HuMoments(cv2.moments(output)).flatten()
        Hi = -np.sign(shape_moments)*np.log10(np.abs(shape_moments))
        ref_im_moments.append(Hi)
        
    return ref_im_moments

def get_reference_images(reference_images):
    ref_imgs = []

    for ref_im in reference_images:
        ref_im = np.load(ref_im).astype(np.uint8)
        
        ref_imgs.append(ref_im)
                
    return ref_imgs

def remove_bad_groups(img_thre, labelled_grps, count, size_thresholds):
    lower_size_thre, upper_size_thre = size_thresholds

    # Draw an image that only contains the groups meeting the size and shape thresholds
    new_grp_img = np.zeros(img_thre.shape)
    group_idxs = []
    for idx, i in enumerate(count):
        # size thresholds determined empirically
        if (i > lower_size_thre) and (i < upper_size_thre):
            # ensure the shape is not oblong
            where_arr = np.array(np.where(labelled_grps == idx))
            aspect = where_arr.max(axis = 1)-where_arr.min(axis = 1)
            if aspect.max()/aspect.min() < 1.2:
                group_idxs.append(idx)
                new_grp_img[np.where(labelled_grps == idx)] = idx
    
    return new_grp_img, group_idxs

# Hu difference formulae
def L1_dif(H1, H2):
    return np.sum(np.abs((1/H1)-(1/H2)))

def L2_dif(H1, H2):
    return np.sum(np.abs(H1-H2))

def L3_dif(H1, H2):
    return np.sum(np.abs(H1-H2)/np.abs(H1))


def get_hu_diffs_for_all_groups(new_grp_img, group_idxs, ref_im_moments):
    # collector for all differences between the group and reference image hu moments
    hu_diffs = []
    for i in group_idxs:
        ref_hu_diffs = []
        # The shape of the group, cropped to its max and min
        pixels = np.array(np.where(new_grp_img == i))
        x_max, y_max = np.max(pixels, axis = 1)
        x_min, y_min = np.min(pixels, axis = 1)
        shape = new_grp_img[x_min-1:x_max+1, y_min-1:y_max+1]

        # Find the contour of the shape
        cont = cv2.findContours(shape.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # plot the contour
        output = np.zeros(shape.shape, dtype=np.uint8)
        cv2.drawContours(output, cont[0], -1, 255)

        # Find the log normalized hu moment of the contour
        shape_moments = cv2.HuMoments(cv2.moments(output)).flatten()
        Hi = -np.sign(shape_moments)*np.log10(np.abs(shape_moments))

        # find the difference between moments
        for ref_Hi in ref_im_moments:
            ref_hu_diffs.append(L2_dif(Hi, ref_Hi))

        # keep the lowest hu difference for the shape
        hu_diffs.append(np.min(ref_hu_diffs))
    return hu_diffs

def get_IoU_for_all_groups(new_grp_img, group_idxs, ref_imgs):
    # collector for all differences between the group and reference image hu moments
    IoUs = []
    for i in group_idxs:
        ref_IoUs = []
        # The shape of the group, cropped to its max and min
        pixels = np.array(np.where(new_grp_img == i))
        x_max, y_max = np.max(pixels, axis = 1)
        x_min, y_min = np.min(pixels, axis = 1)
        shape = new_grp_img[x_min:x_max, y_min:y_max]

        # find the difference between moments
        for ref_img in ref_imgs:
            shape_resized = cv2.resize(shape, np.flip(ref_img.shape[:2]), cv2.INTER_AREA).astype(np.uint8)
            I = np.where((ref_img)&(shape_resized))[0].size
            U = np.where((ref_img)|(shape_resized))[0].size
            IoU = I/U
            ref_IoUs.append(IoU)

        # keep the lowest hu difference for the shape
        IoUs.append(np.max(ref_IoUs))
    return IoUs

def get_plus_centerpoints(reference_images, full_img, thresholds, size_thresholds, slide_brightness_estimate=0.5, show_mask_for_marker=False, increase_contrast_for_markers = True):
    centerpoints = []

    img = sharpen_image_for_marker_detection(full_img, slide_brightness_estimate, do_nothing = not increase_contrast_for_markers)
    # img = full_img.copy()

    ref_imgs = get_reference_images(reference_images)

    # run for each provided threshold
    for thre in thresholds:
        # threshold image
        img_thre = np.where(img>thre, 255, 0).astype(np.uint8)
        # show(img_thre)

        # Group contiguous sets of pixels
        labelled_grps = scipy.ndimage.label(img_thre, structure=np.ones((3,3)), output=None)[0]
        _, count = np.unique(labelled_grps, return_counts = True)

        # Draw an image that only contains the groups meeting the size and shape thresholds
        new_grp_img, group_idxs = remove_bad_groups(img_thre, labelled_grps, count, size_thresholds)
        if show_mask_for_marker:
            show(new_grp_img)

        IoUs = get_IoU_for_all_groups(new_grp_img, group_idxs, ref_imgs)
        # print(IoUs)

        # Sort IoUs
        sorted_IoUs = np.argsort(IoUs)
        # print(sorted_IoUs)
        # show(np.where(new_grp_img == group_idxs[sorted_IoUs[-1]], 255, 0))
        # show(np.where(new_grp_img == group_idxs[sorted_IoUs[-2]], 255, 0))
        # Get the centerpoints of the 2 most similar shapes to the references
        if sorted_IoUs.size:
            mask = np.where(new_grp_img == group_idxs[sorted_IoUs[-1]], 255, 0)
            c1 = scipy.ndimage.center_of_mass(mask, labels=None, index=None)
            centerpoints.append(c1)
        if sorted_IoUs.size>1:
            mask = np.where(new_grp_img == group_idxs[sorted_IoUs[-2]], 255, 0)
            c2 = scipy.ndimage.center_of_mass(mask, labels=None, index=None)
            centerpoints.append(c2)

    # print(centerpoints)
    return centerpoints

def determine_ideal_cluster_assignments(max_possible_num_clusters, centerpoints):

    # initialize collectors
    sse_list = []
    cluster_assignments = []
    centroids = []
    
    clust_range = range(1,max_possible_num_clusters)

    # check intra cluster error for each number of clusters 
    for i in clust_range:
        km = KMeans(n_clusters = i)
        km.fit(centerpoints)
        sse_list.append(km.sse)
        cluster_assignments.append(km.assigned_centroids)
        centroids.append(km.centroids)

    # Get the ideal number of clusters using elbow point method of log error
    num_clusters_index = get_elbow_point(sse_list, clust_range)
    # centroids = centroids[num_clusters_index]
    cluster_assignments = cluster_assignments[num_clusters_index]

    return cluster_assignments

def get_mean_marker_centerpoints(centerpoints, cluster_assignments, as_int:bool = False):
    
    data = np.array(centerpoints)

    # select the top 2 clusters with the greatest number of occurances
    val, count = np.unique(cluster_assignments, return_counts  = True)
    selected_clusters = val[np.argsort(count)[-2:]]

    # get centroids by getting mean of all points within cluster
    centroid1 = np.mean(data[np.where(cluster_assignments == selected_clusters[0])[0]], axis = 0)
    centroid2 = np.mean(data[np.where(cluster_assignments == selected_clusters[1])[0]], axis = 0)

    if as_int:
        # get centroids as int
        return centroid1.astype(int), centroid2.astype(int)
    else:
        # get centroids as float
        return centroid1, centroid2

def show_markers_with_centerpoint(lr_img_full, c1, c2):
    img = lr_img_full.copy()
    img[c1[0], c1[1]] = 0
    img[c2[0], c2[1]] = 0
    show(img[c1[0]-100:c1[0]+100, c1[1]-100:c1[1]+100])
    show(img[c2[0]-100:c2[0]+100, c2[1]-100:c2[1]+100])

def determine_slide_orientation(image, c1, c2, debug = False):
    img_shape = image.shape
    # if points are separated vertically, they are on the left or the right
    if np.argmax(np.abs(c1-c2)) == 0:
        if (img_shape[1] - c1[1] > c1[1]) and (img_shape[1] - c2[1] > c2[1]):
            plus_orientation = 'left'
        elif (img_shape[1] - c1[1] < c1[1]) and (img_shape[1] - c2[1] < c2[1]):
            plus_orientation = 'right'
        else: 
            print("error in detecting the orientation of the '+' shaped markers. Defaulting to left orientation")
            plus_orientation = 'left'
    
    # if points are separated horizontally, they are on the top or bottom
    else:
        if (img_shape[0] - c1[0] > c1[0]) and (img_shape[0] - c2[0] > c2[0]):
            plus_orientation = 'up'
        elif (img_shape[0] - c1[0] < c1[0]) and (img_shape[0] - c2[0] < c2[0]):
            plus_orientation = 'down'
        else: 
            print("error in detecting the orientation of the '+' shaped markers. Defaulting to left orientation")
            plus_orientation = 'left'


    if plus_orientation == 'left':
        if c1[0] < c2[0]:
                c1, c2 = c2, c1
        if debug:
            print('left')
    elif plus_orientation == 'right':
        if c1[0] > c2[0]:
                c1, c2 = c2, c1
        if debug:
            print('right')
    elif plus_orientation == 'up':
        if c1[1] > c2[1]:
                c1, c2 = c2, c1
        if debug:
            print('up')
    elif plus_orientation == 'down':
        if c1[1] < c2[1]:
                c1, c2 = c2, c1
        if debug:
            print('down')

    return plus_orientation, c1, c2


def find_rectangle_corners(image, c1, c2, orientation, display = False):
    image = image.copy()
    if orientation in ['up', 'down']:#,'right', 'down']:
        c1, c2 = c2, c1
    dxy = c2 - c1
    topleft = c2 + (dxy*0.15) - np.flip(dxy)*.1
    bottomleft = c1 - (dxy*0.15) - np.flip(dxy)*.1
    dxy = topleft - bottomleft
    topright = topleft + -np.flip(dxy)*2.15
    bottomright = bottomleft + -np.flip(dxy)*2.15

    vertices = [topleft, topright, bottomright, bottomleft, topleft]

    if display:
        for i in range(4):
            point1 = vertices[i].astype(int)
            point2 = vertices[i+1].astype(int)
            cv2.line(image, np.flip(point1), np.flip(point2), (255), 5)
        show(image)
    
    vertices = np.array(vertices)
    x1, y1 = vertices.min(axis = 0)
    x2, y2 = vertices.max(axis = 0)
    bbox = [x1, y1, x2, y2]
    bbox = np.round(bbox).astype(int)
    return bbox

def check_for_overlapping_bboxes(bboxes):
    overlapping = []
    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(bboxes):
            if i <= j:
                continue
            if do_bboxes_overlap(bbox1, bbox2):
                overlapping.append([i,j])

    return overlapping


def do_bboxes_overlap(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    # # if rectangle has area 0, no overlap
    # if l1.x == r1.x or l1.y == r1.y or r2.x == l2.x or l2.y == r2.y:
    #     return False
     
    # If one rectangle is on left side of other
    if xmin1 > xmax2 or xmin2 > xmax1:
        return False
 
    # If one rectangle is above other
    if ymin1 > ymax2 or ymin2 > ymax1:
        return False
 
    return True

def get_final_tissue_masks(contours, gradient):
    # get masks that meet the size and centroid location requirements
    potential_masks, potential_mask_bboxes = get_potential_masks(contours, gradient)

    # check if mask bboxes overlap
    overlapping_idxs = check_for_overlapping_bboxes(potential_mask_bboxes)

    # combine masks that overlap
    final_masks = combine_overlapping_masks_to_get_final_masks(potential_masks, potential_mask_bboxes, overlapping_idxs)

    return final_masks

    
def get_potential_masks(contours, gradient, display = False):

    img_size = gradient.size
    centroid_limits_y = np.array([gradient.shape[0]*.1, gradient.shape[0]*.9])
    centroid_limits_x = np.array([gradient.shape[1]*.1, gradient.shape[1]*.9])

    selected_contours = []
    potential_masks = []
    potential_mask_bboxes = []

    # Find size of each contour
    sizelist = np.array([cv2.contourArea(contour) for contour in contours])
    meets_size_threshold = np.where((sizelist > img_size/40) & (sizelist < img_size/3))[0]
    for i in meets_size_threshold:
        selected_contours.append(contours[i])

    # Assume tissue will be less than 1/3 the area of the slide but more than 1/40 the area.
    # Selects the largest contour fulfilling this criterion
    biggest_contour = None
    hu_dists = []
    for i in selected_contours:            
        potential_mask = np.zeros(gradient.shape[0:2], dtype = np.uint8)
        cv2.drawContours(potential_mask,[i],0,255,-1)

        # ignore contours with centroids at the edge of the slide
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if not ((centroid_limits_x[0] < cx < centroid_limits_x[1]) & (centroid_limits_y[0] < cy < centroid_limits_y[1])):
                continue
        
        potential_masks.append(potential_mask)
        miny, minx = np.array(i).min(axis=0).flatten()
        maxy, maxx = np.array(i).max(axis=0).flatten()
        bbox = [minx, miny, maxx, maxy]
        potential_mask_bboxes.append(bbox)
    if display:
        for i in potential_masks:
            show(i)
    return potential_masks, potential_mask_bboxes
    
    
def combine_overlapping_masks_to_get_final_masks(potential_masks, potential_mask_bboxes, overlapping_idxs):

    # list of indices that have been used    
    used_idxs = np.empty(0)
    mask_idxs = np.ones(len(potential_mask_bboxes), dtype = int)*-1
    final_masks = []

    # Indices of the final masks that are duplicates and need to be deleted
    mark_for_deletion = []

    # print(overlapping_idxs)
    for i, j in overlapping_idxs:
        #list of masks to combine in this loop
        masks_to_combine = []
        # current index of masks
        current_mask_idx = len(final_masks)

        # check if i or j have already been used in a combination
        matching_idxs_i, matching_idxs_j = [], []
        if i in used_idxs:
            matching_idxs_i.append(mask_idxs[i])
        if j in used_idxs:
            matching_idxs_j.append(mask_idxs[j])

        # add corresponding mask to masks to combine if i or j have not been used
        # add the masks index to the used indices list
        if not len(matching_idxs_i): 
            masks_to_combine.append(potential_masks[i])
    
        if not len(matching_idxs_j): 
            masks_to_combine.append(potential_masks[j])

        # combine the matching indices list
        matching_idxs = matching_idxs_i + matching_idxs_j

        # if there are any matching indices 
        # print(matching_idxs)

        # Get the already combined masks to combine again
        # idxs_to_combine = np.empty(0)
        if len(matching_idxs):
            # print(matching_idxs, mask_idxs)
            # idxs_to_combine = np.unique(mask_idxs[np.array(matching_idxs)])
        
            for idx_to_combine in matching_idxs:
                # print(idx_to_combine)
                masks_to_combine.append(final_masks[idx_to_combine])
        
            # be ready to assign the new mask to the index of the old mask that is being combined again
            current_mask_idx = matching_idxs[0]

            # any idxs to combine after the first are changed to the first value and are saved to delete that mask
            for idx in matching_idxs[1:]:
                mask_idxs = np.where(mask_idxs == idx, matching_idxs[0], mask_idxs)
                mark_for_deletion.append(idx)
        
        else:
            # mark that ith and jth entry in mask indices are for the current mask index
            mask_idxs[i] = current_mask_idx
            mask_idxs[j] = current_mask_idx

        # combine all masks 
        # uses bitwise or to mark every pixel with a nonzero value in any mask as 255
        final_mask = combine_masks(masks_to_combine)

        # Append the new mask or replace one that was used to make it
        if current_mask_idx == len(final_masks):
            final_masks.append(final_mask)
        else:
            final_masks[current_mask_idx] = final_mask

        # mark the i and j values as used
        used_idxs = np.append(used_idxs, [i, j])
        used_idxs = np.unique(used_idxs)

    return final_masks

def combine_masks(masks_to_combine):
    masks_to_combine = np.stack(masks_to_combine, axis = 0).astype(bool)
    return np.bitwise_or.reduce(masks_to_combine, axis = 0).astype(np.uint8)*255


def upscale_imgs_back_to_orig(input_masks, img_full, img_reduced = None, bbox = None):
    masks = []
    for i in input_masks:
        if type(img_reduced) != type(None):
            mask_zero = np.zeros(img_full.shape[0:2], dtype = np.uint8)
            mask = cv2.resize(i, np.flip(img_reduced.shape[0:2]), interpolation = cv2.INTER_NEAREST)
            mask_zero[bbox[0]:bbox[2], bbox[1]:bbox[3]] = mask
        else: 
            mask_zero = cv2.resize(i, np.flip(img_full.shape[0:2]), interpolation = cv2.INTER_NEAREST)
        masks.append(mask_zero)
    return masks


def order_the_masks(masks, orientation_of_markers = 'left'):
    # standardize the orientation name
    o = fix_orientation_name(orientation_of_markers)

    
    # ensure the masks are of the correct shape and number of axes
    masks = np.array(masks)
    if np.ndim == 2:
        masks = masks[np.newaxis]

    # values for ordering masks
    metric = []

    # sort masks in increasing order from closer to farther from the markers
    if o == 'left':
        for mask in masks:
            metric.append(np.array(np.where(mask)).min(axis = 1)[1])
        masks = masks[np.argsort(metric)]
    if o == 'right':
        for mask in masks:
            metric.append(np.array(np.where(mask)).max(axis = 1)[1])
        masks = masks[np.flip(np.argsort(metric))]
    if o == 'up':
        for mask in masks:
            metric.append(np.array(np.where(mask)).min(axis = 1)[0])
        masks = masks[np.argsort(metric)]            
    if o == 'down':
        for mask in masks:
            metric.append(np.array(np.where(mask)).max(axis = 1)[0])
        masks = masks[np.flip(np.argsort(metric))]
    return masks

def fix_orientation_name(orientation_of_markers):
    o = orientation_of_markers
    if o.lower() in ['left']:
        o = 'left'
    if o.lower() in ['right']:
        o = 'right'
    if o.lower() in ['top', 'up', 'above', 'upper']:
        o = 'up'
    if o.lower() in ['bottom', 'down', 'below', 'lower']:
        o = 'down'
    return o

def convert_to_point_matrix(coord):
    starting_coord = np.ones((3,1), dtype = coord.dtype)
    starting_coord[:2] = coord[:,None]
    return starting_coord

# make_new_coordinates
def determine_transform_to_new_coords(c1, c2):
    # convert c1 to 3x1 matrix
    starting_coord = convert_to_point_matrix(c1)

    # get transformation components
    # translation
    translation = np.eye(3, dtype = c1.dtype)
    translation[:2,-1] = -c1
    #scaling
    norm_diff = 1/np.linalg.norm(c1-c2)
    scaling = np.eye(3, dtype = c1.dtype)*norm_diff
    scaling[-1,-1] = 1
    # rotation
    x = c2-c1
    theta = np.arctan2(x[0],x[1])
    rotation = np.eye(3, dtype = c1.dtype)
    rotation[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta) ]])

    # combine transforms
    transform_matrix = rotation@scaling@translation
    return transform_matrix

def apply_transform(transform_matrix, coordinate):
    # transform given point
    output = transform_matrix@convert_to_point_matrix(coordinate)

    # account for weird tiny outputs due to floating point math
    output = np.where(np.isclose(output, np.zeros(1), rtol=1e-08), 0, output)

    return output[:2]

def get_tissue_bboxes_in_new_coordinate_plane(masks, transform_matrix):
    bboxes = []
    for mask in masks:
        # Find points at which to apply the transform
        X, Y = np.where(mask)
        
        # a third axis of ones is needed to multiply by the transform matrix
        Z = np.ones(len(X))
        
        # find the coordinates in the new coordinate space
        coords = np.einsum('ji, mni -> jmn', transform_matrix, np.dstack([X,Y,Z]))
        coords = coords[:2].reshape(2,-1)

        # find the min and max of x and y to get a bounding box
        x_min, y_min = coords.min(axis = 1)
        x_max, y_max = coords.max(axis = 1)
        bbox = [x_min, y_min, x_max, y_max]

        bboxes.append(bbox)
    return bboxes

def save_bboxes_as_csv(bboxes, image_paths, csv_name = None):
    
    img_names = []
    new_bboxes = []
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.split(img_path)[-1]
        for i in range(len(bboxes[idx])):
            img_names.append(img_name)
            new_bboxes.append(bboxes[idx][i])

    if not csv_name:
        csv_name = '.'.join(image_paths[0].split('.')[:-1])+'.csv'
    else:
        if not csv_name.endswith('.csv'):
            folder_path = os.path.split(image_paths[0])[0]
            csv_name = os.path.join(folder_path, csv_name + '.csv')

    if not os.path.exists(os.path.split(csv_name)[0]):
        csv_name = os.path.join(os.getcwd(),os.path.split(csv_name)[-1])


    df = pd.DataFrame(img_names, columns = ['filenames'])
    headers = ['x_1', 'y_1', 'x_2', 'y_2']
    df2 = pd.DataFrame(np.array(new_bboxes).reshape(-1,4), columns = headers)
    
    df.join(df2).to_csv(csv_name, index = False)
    print(f'Output csv saved at {csv_name}')


def determine_img_paths_to_use(image_paths, img_indices):
    ImagePathException = TypeError('image_paths must be a list of filepaths')
    ImageIndicesException = TypeError('img_indices must be an int, or a list or tuple of ints')

    if type(img_indices) == type(None):
        image_paths_to_use = image_paths
        if type(image_paths_to_use) == tuple:
            image_paths_to_use = list(image_paths_to_use)
        if type(image_paths_to_use) == str:
            image_paths_to_use = [image_paths_to_use]
        if type(image_paths_to_use) != list:
            raise(ImagePathException)
        for i in image_paths_to_use:
            if type(i) != str:
                raise(ImagePathException)


    if type(img_indices) == int:
        image_paths_to_use = image_paths[img_indices]

    if type(img_indices) in [list, tuple]:
        image_paths_to_use = []
        for i in img_indices:
            if type(img_indices) == int:
                image_paths_to_use.append(image_paths[i])
            else:
                raise ImageIndicesException
    else:
        ImageIndicesException

    return image_paths_to_use

def get_tissue_regions_as_bbox(csv_name:str, image_paths:list, ref_imgs, downscale_factor:int = 5, img_indices: int or list = None, slide_brightness_estimate: float = 0.5,
                               show_gradient_img = False, show_mask_for_marker = False, increase_contrast_for_markers = True, threshold_for_tissue = None):

    image_paths = determine_img_paths_to_use(image_paths, img_indices)
    
    full_masks_collector = []
    bboxes_collector = []
    centerpoints_collector = []
    c_collector = []

    for image_path in image_paths:
        # read image
        img_full = io.imread(image_path)
        bounds = img_full.shape[1]//3, img_full.shape[1]*2//3
        img_full = img_full[:,bounds[0]:bounds[1]]
        # img for finding markers
        # cropped_full_img = img_full[:img_full.shape[0]//3]

        # downsize images for speed and smoothing
        # determine new dimensions for downsized image
        # new_dim = np.flip(np.round(np.array(img_full.shape[0:2])/downscale_factor).astype(int))

        # use fewer thresholds when the background contrast is better
        min_thre, max_thre = .75, .9
        thresholds = np.linspace(min_thre, max_thre, num = int(np.round(((max_thre-min_thre)*50)+1)))
        size_thre = [img_full.size//8000, img_full.size//500] # empirically determined

        # get_plus_centerpoints from potential marker contours on thresholded images
        centerpoints = get_plus_centerpoints(ref_imgs, img_full, thresholds, size_thre, slide_brightness_estimate, show_mask_for_marker, increase_contrast_for_markers = increase_contrast_for_markers)
        # centerpoints = get_plus_centerpoints(ref_imgs, cropped_full_img, thresholds, size_thre, slide_brightness_estimate)

        # get the cluster assignments for ideal number of clusters
        cluster_assignments = determine_ideal_cluster_assignments(6, centerpoints)
        # get the centroid of the clusters as the location of the markers
        c1, c2 = get_mean_marker_centerpoints(centerpoints, cluster_assignments, as_int = False)

        # show_centerpoints(c1, c2, img_full)

        # determine the orientation of the slide based on the markers
        orientation, c1, c2 = determine_slide_orientation(img_full, c1, c2)

        # determine the region that may potenntially contain tissue sections
        bbox = find_rectangle_corners(img_full, c1, c2, orientation,  display = False)

        # crop image to potential tissue area
        img_cropped_full = img_full[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        # determine how to downscale the cropped image
        new_dim_cropped = np.flip(np.round(np.array(img_cropped_full.shape[0:2])/downscale_factor).astype(int))

        # downsize the cropped image
        img_cropped_downsized = cv2.resize(img_cropped_full, new_dim_cropped, interpolation = cv2.INTER_AREA)
        img_cropped_downsized = (img_cropped_downsized/img_cropped_downsized.max()).astype(np.float32)

        # median blur image to eliminate spikes
        blur = cv2.medianBlur(img_cropped_downsized,5)

        # convert to LAB colorspace for better edge detection
        img_cropped_downsized_lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)

        # Get gradient of image using HyAB method
        cropped_downsized_hyab_gradient = get_hyab_gradient_img(img_cropped_downsized_lab)
        if show_gradient_img:
            show(cropped_downsized_hyab_gradient)

        # get contours
        img_contours = get_contours(cropped_downsized_hyab_gradient, display = False, threshold_for_tissue = threshold_for_tissue)

        # get masks for downscaled lr img
        cropped_downsized_masks = get_final_tissue_masks(img_contours, cropped_downsized_hyab_gradient)
        cropped_downsized_masks = order_the_masks(cropped_downsized_masks, orientation_of_markers = orientation)

        # upscale masks to full size
        full_masks = upscale_imgs_back_to_orig(cropped_downsized_masks, img_full, img_cropped_full, bbox)
        # small_masks = [cv2.resize(i, new_dim_cropped, interpolation = cv2.INTER_AREA) for i in full_masks]

        # when the '+' markers are on the top of the image, the left is at (0,0) and the right is at (0,1)
        transform_matrix = determine_transform_to_new_coords(c1, c2)
        # print(full_masks)
        bboxes = get_tissue_bboxes_in_new_coordinate_plane(full_masks, transform_matrix)
        # print(bboxes)
        
        show_centerpoints_and_bboxes(img_full, c1, c2, full_masks)

        full_masks_collector.append(full_masks)
        bboxes_collector.append(bboxes)
        centerpoints_collector.append(centerpoints)
        c_collector.append((c1, c2))

    save_bboxes_as_csv(bboxes_collector, image_paths, csv_name)

    return full_masks_collector, bboxes_collector, c_collector, centerpoints_collector

def show_centerpoints(c1, c2, img_full):
    img = img_full.copy()
    c1 = c1.astype(int)
    c2 = c2.astype(int)
    img[c1[0]-10:c1[0]+10, c1[1]-10:c1[1]+10] = [255,0,0]
    img[c2[0]-10:c2[0]+10, c2[1]-10:c2[1]+10] = [255,0,0]
    show(img)

def show_centerpoints_and_bboxes(img_full, c1, c2, full_masks):
    img = img_full.copy()
    c1 = c1.astype(int)
    c2 = c2.astype(int)
    img[c1[0]-10:c1[0]+10, c1[1]-10:c1[1]+10] = [255,0,0]
    img[c2[0]-10:c2[0]+10, c2[1]-10:c2[1]+10] = [255,0,0]

    # draw rectangle around each tissue
    for mask in full_masks:
        xmin, ymin = np.min(np.where(mask), axis = 1)
        xmax, ymax = np.max(np.where(mask), axis = 1)
        img = cv2.rectangle(img, (ymin, xmin), (ymax, xmax), [0,255,0], 5)
    show(img)

def show_processed_tissues(image_paths, all_full_masks, all_centerpoints):
    for i, img_path in enumerate(image_paths):
        img_full = cv2.imread(img_path).astype(np.uint8)
        img_full = img_full[:,:,[2,1,0]]
        bounds = img_full.shape[1]//3, img_full.shape[1]*2//3
        img_full = img_full[:,bounds[0]:bounds[1]]

        full_masks = all_full_masks[i]

        c1, c2 = all_centerpoints[i]

        show_centerpoints_and_bboxes(img_full, c1, c2, full_masks)
