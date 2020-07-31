import os
import random

all_pics = []
fldr = {}

for root, dirs, files in os.walk("images", topdown=False):
    for name in files:
       if name.endswith("jpg"):
           all_pics.append(os.path.join(root, name))
           if root not in fldr:
               fldr[root] = 1

all_pics.sort()


for i, name in enumerate(all_pics):

    tmp = (name.split(".jpg")[0]).split(os.path.sep)[-1]
    if (random.randint(0,10) == 5):
        os.system("cp images/" + tmp + ".jpg val/images")
        os.system("cp labels/" + tmp + ".png val/labels")
    else:
        os.system("cp images/" + tmp + ".jpg train/images")
        os.system("cp labels/" + tmp + ".png train/labels")
    #tmp = name.split("jpg")[0] + "png"
    ##print(tmp)
    #os.rename(name, tmp)
