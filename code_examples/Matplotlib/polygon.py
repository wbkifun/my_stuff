import numpy as np
import matplotlib
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt


#fig, ax = plt.subplots()
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1)


# Circle
center, radius = (0.4,0.3), 0.1
ax.add_patch( Circle(center, radius, fill=False, hatch='/') )


# Polygon
points = [(0.3,0.2), (0.4,0.3), (0.38,0.4), (0.25,0.45), (0.2,0.35)]
ax.add_patch( Polygon(points, fill=True, color='b', alpha=0.4) )


plt.show()
