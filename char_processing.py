import glob
import cv2
import os
import pathlib
import numpy as np
    
def processed_label(label):
    label = os.path.basename(label)
    label = os.path.splitext(label)[0]
    label = label[:-2]
    label = label.lower()
    return label

def char_tokenizer(post_process,label):
    imgs = post_process
    label = processed_label(label)
    # if len(imgs) == 1:
    #     # has only single largest images predicted that it would be the 
    #     return imgs,[label]
    if len(imgs) == len(label):
        return imgs,[i for i in label]
    elif len(imgs) != len(label):
        print("passed")
        character_deficit = np.abs(len(imgs) - len(label))
        return None,None
    # elif len(imgs) != len(label):
    #     # intuitively multicharcater might be in the higher text
    #     character_deficit = np.abs(len(imgs) - len(label))
        

if __name__ == "__main__":
    print(processed_label("test-0.jpg"))
    for i in processed_label("test-0.jpg"):
        print(i)