import os

all_pics = []
fldr = {}

for root, dirs, files in os.walk("labels/", topdown=False):
    for name in files:
       if name.endswith(".png"):
           all_pics.append(os.path.join(root, name))
           if root not in fldr:
               fldr[root] = 1

all_pics.sort()


for i, name in enumerate(all_pics):
    print(name)
    os.rename(name, 'labels/' + (8 - len(str(i)))*"0" + str(i) + ".png")

all_pics = []
fldr = {}

for root, dirs, files in os.walk("images/", topdown=False):
    for name in files:
       if name.endswith("jpg"):
           all_pics.append(os.path.join(root, name))
           if root not in fldr:
               fldr[root] = 1

all_pics.sort()


for i, name in enumerate(all_pics):
    print(name)
    os.rename(name, 'images/' + (8 - len(str(i)))*"0" + str(i) + ".jpg")
