from PIL import Image
import os

path = "/Users/kasper/Documents/GitHub/Master/images500x500/Steel"

for image_path in os.listdir(path):
    #image = "/Users/kasper/Documents/GitHub/Master/001test.png"
    input_path = os.path.join(path, image_path)
    new_name = "steel" + image_path
    output_path = os.path.join(path, new_name)
    #print(image_path)
    #print(new_name)

    os.rename(input_path, output_path)
    