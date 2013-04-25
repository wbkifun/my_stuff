from pylab import *
from math import log


H = 500
iterations = 20
ColorList = []

def mandelbrot(real, imag):
    #z = complex(-0.726895347709114071439, 0.188887129043845954792)
    z = complex(0,0)
    c = complex(real, imag)
    dropOutLevel = 0
    for i in range(0, iterations):
        if abs(z) < 2:
            z = z**2 + c
            dropOutLevel += 1
        else:
            break

    z = z**2 + c
    z = z**2 + c

    return dropOutLevel,z


def interpolateC(color,endColor,left):
    if left == 0:
        return color
    rStart = color[0]
    gStart = color[1]
    bStart = color[2]
    rEnd = endColor[0]
    gEnd = endColor[1]
    bEnd = endColor[2]
    rDiff = rEnd-rStart
    gDiff = gEnd-gStart
    bDiff = bEnd-bStart
    rInc = float(rDiff)/left
    gInc = float(gDiff)/left
    bInc = float(bDiff)/left
    #print (str(left))

    return [rStart+rInc,gStart+gInc,bStart+bInc]


def colorList():
    cols = []
    points = [[0,0,0],[ 0.988235294118 , 0.760784313725 , 0.0 ],[0,0,0]]
    eachLength = H/(len(points)-1)
    howmanytogo = eachLength

    for a in range(1,len(points)):
        cols.append(points[a-1])
        for i in range (0,eachLength):
            cols.append(interpolateC(cols[-1],points[a],howmanytogo))
            howmanytogo += -1

        howmanytogo = eachLength
    global ColorList
    ColorList = cols


def toColor(a):
    a=a*100
    a = int(a)%H
    try:
        return ColorList[int(a)]
    except IndexError:
        print a
        print len(ColorList)
        return ColorList[0]


def v(z,n):
    try:
        x = n + 1 - log2(log2(abs(z)))/log2(2)
        return x
    except ValueError:
        print(str(z))
    return 0


colorList()
wDensity = 500.0
hDensity = 500.0
x_ords = []
y_ords = []
colors = []
left = float(raw_input("What is the leftmost value? "))
right = float(raw_input("What is the rightmost value? "))
bottom = float(raw_input("What is the bottommost value? "))
top = float(raw_input("What is the topmost value? "))
print "Calculating... "
w = right - left
h = top - bottom
wGap = w/wDensity
hGap = h/hDensity
wPercentageGap = w/100
a = left
perc = 0
print "0%"
while a < right:
    b = bottom
    while b < top:
        x_ords.append(a)
        y_ords.append(b)
        n, z = mandelbrot(a,b)
        if (n == iterations):
            col = [0.0,0.0,0.0]
        else:
            col = toColor(v(z,n))
        #print col
        colors.append(col)
        b += hGap
    a += wGap
    curPerc = (right-a)/w * 100
    if curPerc > perc+10:
        perc += 10
        print str(perc),"%"
    if curPerc > perc+5:
        perc += 5
        print str(perc),"%"

print "Rendering... "
autoscale(tight=True)
scatter(x_ords, y_ords, s=2, c=colors, linewidths=0)
F = gcf()
F.set_size_inches(wDensity/60,hDensity/60)
#
F.savefig('GoldAwesomeSmooth.png')
