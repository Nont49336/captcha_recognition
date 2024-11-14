import numpy as np
import cv2
# import for inference
import matplotlib.pyplot as plt

def img_preprocessing(img_file_path):
    '''
    This process handling the image preprocessing basically process 
    the image as a whole to get rid of line and obscure image
    input: img_path
    return: preprocessed image clean images
    '''
    img = cv2.imread(img_file_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = np.uint8(np.where((gray==0),255,0))
    img = cv2.inpaint(img,mask,inpaintRadius=3,flags=cv2.INPAINT_TELEA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bw_mask = np.uint8(np.where((gray==255),0,255))
    kernel = np.ones((3,3),np.uint8) 
    img = cv2.morphologyEx(bw_mask,cv2.MORPH_OPEN,kernel=kernel)
    return img
    
def img_postprocessing(img,char_padding=1):
    '''
    Processing function for image tokenizer
    input: image
    output: a list of cropped binary-image sorted from left to right
    '''
    tmp_res = cv2.connectedComponentsWithStats(img,8,cv2.CV_32S)
    (numLabels,labels,stats,centroids) = tmp_res
    
    connected_pos=[]
    saved_part = []
    connected_region_area = []
    output = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            # text = "examining component {}/{} (background)".format(
            #     i + 1, numLabels)
            continue
        # otherwise, we are examining an actual connected component
            # text = "examining component {}/{}".format( i + 1, numLabels)
        # print a status message update for the current connected
        # component
        # print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # output = img.copy()
        
        # connected_region_area.append((w*h))
        connected_pos.append([x,y,w,h])

        sorted_roi = sorted(connected_pos, key=lambda x: x[0])
        
        saved_part = []
        connected_region_area = []
        for i in sorted_roi:
            (x,y,w,h) = i
            roi = img[y:y+h,x:x+w]
            connected_region_area.append((w*h))
            roi = cv2.copyMakeBorder(roi,char_padding,char_padding,char_padding,char_padding,borderType=cv2.BORDER_CONSTANT,value=0)
            saved_part.append(img[y-char_padding:y+h+char_padding,x-char_padding:x+w+char_padding])
    
    return saved_part,connected_region_area

# def mat_show(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # plt.figure(figsize=(12, 6))
#     plt.imshow(rgb_image)
#     plt.axis('off')  # Turn off axis
#     plt.show()

def fill_sandwiched_pixels(img, mask):
    # Create a copy of the image to modify
    img_filled = img.copy()

    # Get image dimensions
    height, width, _ = img.shape

    # Iterate through all pixels
    for y in range(1, height - 1):  # Start from 1 and go to height-1 to avoid edges
        for x in range(1, width - 1):  # Start from 1 and go to width-1 to avoid edges
            if mask[y, x] != 0:  # Check if this pixel was originally black
                # Initialize variables to hold surrounding pixel values
                pixel_above = img[y - 1, x]
                pixel_below = img[y + 1, x]
                pixel_left = img[y, x - 1]
                pixel_right = img[y, x + 1]
                pixel_top_left = img[y - 1, x - 1]
                pixel_bottom_right = img[y + 1, x + 1]
                pixel_top_right = img[y - 1, x + 1]
                pixel_bottom_left = img[y + 1, x - 1]

                # Check the conditions for each pair of surrounding pixels
                # 1. Top and Bottom
                if np.all(pixel_above != [255, 255, 255]) and np.all(pixel_below != [255, 255, 255]):
                    new_pixel = pixel_above
                    img_filled[y, x] = new_pixel
                
                # 2. Left and Right
                elif np.all(pixel_left != [255, 255, 255]) and np.all(pixel_right != [255, 255, 255]):
                    new_pixel = pixel_right
                    img_filled[y, x] = new_pixel
                
                # 3. Top-Left and Bottom-Right
                elif np.all(pixel_top_left != [255, 255, 255]) and np.all(pixel_bottom_right != [255, 255, 255]):
                    new_pixel = pixel_top_left
                    img_filled[y, x] = new_pixel
                
                # 4. Top-Right and Bottom-Left
                elif np.all(pixel_top_right != [255, 255, 255]) and np.all(pixel_bottom_left != [255, 255, 255]):
                    new_pixel = pixel_bottom_left
                    img_filled[y, x] = new_pixel

    return img_filled

def img_preprocessing_cleon(img_file_path,show=False):
    '''
    This process handling the image preprocessing basically process 
    the image as a whole to get rid of line and obscure image
    input: img_path
    return: preprocessed image clean images
    '''
    img = cv2.imread(img_file_path)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])  # Define the range for black pixels
    mask = cv2.inRange(img, lower_black, upper_black)

    img[mask != 0] = [255, 255, 255]
    img_filled = fill_sandwiched_pixels(img, mask)
    if show:
        cv2.imshow("color_filled:",img_filled)
        cv2.waitKey(0)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # mask = np.uint8(np.where((gray==0),255,0))
    # img = cv2.inpaint(img,mask,inpaintRadius=3,flags=cv2.INPAINT_TELEA)
    img_filled = cv2.cvtColor(img_filled,cv2.COLOR_BGR2GRAY)
    bw_mask = np.uint8(np.where((img_filled==255),0,255))
    if show:
        cv2.imshow("bw_mask:",img_filled)
        cv2.waitKey(0)
    # kernel = np.ones((3,3),np.uint8) 
    # img = cv2.morphologyEx(bw_mask,cv2.MORPH_OPEN,kernel=kernel)
    return bw_mask


if __name__ == "__main__":

    sample_pth = "D:/CS4243_miniproj/captcha_img/train/00hgi3n7-0.png"
    # img = img_preprocessing(sample_pth)
    img = img_preprocessing_cleon(sample_pth)
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    post,area = img_postprocessing(img)
    # cv2.imshow("new_test",img)
    # cv2.waitKey(0)

    