
from lxml import etree
import numpy as np
import pickle
from PIL import Image, ImageDraw


### cd platgesbcn2021_fine/images; for a in $(ls useful/*.JPG); do rm ../../platgesbcn2021_wide/images/$a; done ###
### for a in $(ls useful/*.JPG); do rm ../../platgesbcn2021_wide/images/${a/.JPG/}.*.pkl; done ###


#ANNOTATIONS_ROOT    = "/mnt/c/Users/Ignasi/Downloads/platgesbcn2021_wide/"
ANNOTATIONS_ROOT    = "/mnt/c/Users/Ignasi/Downloads/platgesbcn2021_fine/"
ANNOTATIONS_NAME    = "annotations.xml"
ANNOTATIONS_PATH    = f"{ANNOTATIONS_ROOT}/{ANNOTATIONS_NAME}"

#IMAGES_ROOT         = "/mnt/c/Users/Ignasi/Downloads/platgesbcn2021_wide/images/useful/"
IMAGES_ROOT         = "/mnt/c/Users/Ignasi/Downloads/platgesbcn2021_fine/images/useful/"

INTEREST_LABELS = {"sorra", "aigua"}


def save_classes(classes, img_path):
    classes = {value : key for key, value in classes.items()}
    with open(f'{img_path}.classes.pkl', 'wb') as f:
        pickle.dump(classes, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_mask(mask, img_path):
    with open(f'{img_path}.segments.pkl', 'wb') as f:
        pickle.dump(mask, f)

def poly2mask(coordinates, width, height):
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(coordinates, outline=1, fill=1)
    mask = np.array(img)
    return mask

def main(annotations_path, outputs_root, interest_labels=None):
    classes = dict()

    tree = etree.parse(annotations_path)
    root = tree.getroot()

    images = root[2:] # The first 2 elements are not annotations
    for image in images:
        width = int(image.get('width'))
        height = int(image.get('height'))
        name = image.get('name')
        mask = None

        elements = 0

        for polygon in image:
            label = polygon.get('label')
            if (interest_labels is not None) and (label not in interest_labels) : continue
            if label not in classes.keys() : classes[label] = 1 << len(classes.keys())

            coordinates = [tuple(int(float(coor)) for coor in pt.split(',')) for pt in polygon.get('points').split(';')]
            if mask is None:
                mask = poly2mask(coordinates, width, height) * classes[label]
            else:
                mask = mask | (poly2mask(coordinates, width, height) * classes[label])
            
            elements += 1
        
        if elements > 0:
            save_classes(classes, f"{outputs_root}/{name.rsplit('.', 1)[0]}")
            save_mask(mask, f"{outputs_root}/{name.rsplit('.', 1)[0]}")



if __name__ == "__main__":
    main(ANNOTATIONS_PATH, IMAGES_ROOT, INTEREST_LABELS)
