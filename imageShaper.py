from PIL import Image
import os

path = "/Users/eier/GitHub/Master/testimages/alu_masks/"
out_path = "/Users/eier/GitHub/Master/testimages/alu_cropped_masks/"

for image_path in os.listdir(path):
    #image = "/Users/kasper/Documents/GitHub/Master/001test.png"
    input_path = os.path.join(path, image_path)

    im = Image.open(input_path)
    topLeft_x = 0
    topLeft_y = 4
    bottomRight_x = 500
    bottomRight_y = 504

    cropped_image = im.crop((topLeft_x, topLeft_y, bottomRight_x, bottomRight_y))

    fullpath = os.path.join(out_path, image_path)
    cropped_image.save(fullpath)
    