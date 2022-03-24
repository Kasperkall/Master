from PIL import Image
import os

path = "/Users/kasper/Documents/GitHub/Master/images500x500/Steel"

"""
for image_path in os.listdir(path):
    #image = "/Users/kasper/Documents/GitHub/Master/001test.png"
    input_path = os.path.join(path, image_path)
    new_name = "steel" + image_path
    output_path = os.path.join(path, new_name)
    #print(image_path)
    #print(new_name)

    os.rename(input_path, output_path)"""
    

#path = "/Users/kasper/Documents/GitHub/Master/testimages/alu_masks/"
#out_path = "/Users/kasper/Documents/GitHub/Master/testimages/alu_cropped_masks/"


path = "/Volumes/G-DRIVE/double-scan-v3/"

for image_path in os.listdir(path):
    number = image_path[-4:]
    image_path = image_path+ "/gt_scan.png"
    input_path = os.path.join(path, image_path)
    #print("path",input_path)
    output_path = "/Volumes/G-DRIVE/scan-v4/ground-truth/"

    full_path = os.path.join(output_path, "img" + number + ".png")
    
    im = Image.open(input_path)
    width, height = im.size
    #print("w",width)
    #print("h",height)
    topLeft_x = 300
    topLeft_y = 270
    bottomRight_x = 800
    bottomRight_y = 770

    cropped_image = im.crop((topLeft_x, topLeft_y, bottomRight_x, bottomRight_y))
    
    #fullpath = os.path.join(output_path, full_path)
    #print(full_path)
    cropped_image.save(full_path)



"""
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
    """
    