# get the bounding boxes for tissue sections present in given images
exec(open("./register slide to HR image defs.py").read())

# the '+' shaped markers on the slide are at (0,0) and (0,1) for 
# the left and right respectively when the markers oriented upwards 


# Define Parameters
# -----------------------------------------------------------------------------
downscale_factor = 5  # larger numbers will reduce noise and fine background patterns, but may reduce detail of image outlines

image_paths = [r".\test_imgs\MicrosoftTeams-image.png",
    r".\test_imgs\MicrosoftTeams-image (4).png",
    r".\test_imgs\MicrosoftTeams-image (4)_cropped.png",
    r".\test_imgs\MicrosoftTeams-image (5).png",
    r".\test_imgs\MicrosoftTeams-image (5)_cropped.png",
    r".\test_imgs\MicrosoftTeams-image (6).png",
    r".\test_imgs\MicrosoftTeams-image (7).png",
    r".\test_imgs\MicrosoftTeams-image (8).png",
    r".\test_imgs\MicrosoftTeams-image (9).png"]     
           
# Bounding box information is saved to this csv in the same folder as the images
name_of_output_csv = 'output.csv'
# -----------------------------------------------------------------------------

# Reference image of a plus-shaped mask
# You do not need to change this
ref_imgs = [r".\reference-plus-sign1.npy",
            r".\reference-plus-sign2.npy"]

# get bboxes as csv
full_masks, bboxes, centerpoints, crude_centerpoints = get_tissue_regions_as_bbox(name_of_output_csv, image_paths, ref_imgs, downscale_factor)