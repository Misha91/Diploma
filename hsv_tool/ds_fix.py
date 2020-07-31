import os

all_pics = []
fldr = {}

for root, dirs, files in os.walk("labels", topdown=False):
    for name in files:
       if name.endswith("jpg"):
           all_pics.append(os.path.join(root, name))
           if root not in fldr:
               fldr[root] = 1

all_pics.sort()


for i, name in enumerate(all_pics):
    tmp = name.split("jpg")[0] + "png"
    #print(tmp)
    os.rename(name, tmp)
