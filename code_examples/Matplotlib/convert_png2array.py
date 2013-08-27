from __future__ import division
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint


# read image file
img = Image.open("two_girls.png")

# convert image to numpy array
arr = numpy.array(img)
pprint(arr)

# plot the numpy array
plt.imshow(arr)
plt.show()
