#!/usr/bin/env python

import Image
import ImageDraw

SIZE = 128
image = Image.new("L", (SIZE, SIZE))
d = ImageDraw.Draw(image)

c = 0.5 + 0.2j
for x in range(SIZE):
	for y in range(SIZE):
		re = (x * 2.0 / SIZE) - 1.0
		im = (y * 2.0 / SIZE) - 1.0
		
		z=re+im*1j
		
		for i in range(128):
			if abs(z) > 2.0: break
			z = z * z + c
		d.point((x, y), i * 2)

image.save(r"./julia.png", "PNG")
