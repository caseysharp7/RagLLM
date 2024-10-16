import json
import os

with open('/home/sharpcj/cuda_links.json', 'r') as file:
    links = json.load(file)

# print(os.path.abspath('O_2D_array_device_allocation.cu'))

for i in links:
    one = i["2D_array_device_allocation"]
    two = i["O_2D_array_device_allocation"]
    desc = i.get("Link")

    print(f"Linking {one} with {two}")
    print(f"Description: {desc}")

    # if os.path.exists("/home/sharpcj/my_cuda/unoptimized/memory/2D_array_allocation.cu"):
    with open(one, 'r') as file_1, open(two, 'r') as file_2:
        code_1 = file_1.read()
        code_2 = file_2.read()

        print(f"file 1: {code_1}")
        print(f"file 2: {code_2}")
