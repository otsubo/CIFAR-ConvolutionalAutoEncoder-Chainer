import deepdish as dd
from skimage import io
import os

d = dd.io.load("./0000000_tactileColorR.h5")
#os.mkdir("./test")
for i in range(len(d)):
    image = d[i]
    os.makedirs("./test/10%s" % i, exist_ok=True)
    io.imsave("./test/10%s/image.png" % i , image)
