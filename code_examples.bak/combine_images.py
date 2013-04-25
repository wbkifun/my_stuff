from __future__ import division
import Image, ImageDraw, ImageFont
import sys


N = 30
tmax = 1000
tgap = 1


for tstep in xrange(tgap, tmax+1, tgap):
    print 'tstep=\t%d/%d (%1.2f %s)\r' % (tstep, tmax, tstep/tmax*100, '%'),
    sys.stdout.flush()

    im1 = Image.open('png_pole/%.6d.png' % (tstep))
    im2 = Image.open('png_pole_modified/%.6d.png' % (tstep))

    cim = Image.new('RGB', (1600,800))
    cim.paste(im1, (0,0))
    cim.paste(im2, (800,0))

    '''
    font = ImageFont.truetype('/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf', 30)
    draw = ImageDraw.Draw(cim)
    draw.text((50,10), 'N=%d' % N1, font=font)
    draw.text((850,10), 'N=%d' % N2, font=font)
    '''

    cim.save('png_pole_compare/%.6d.png' % (tstep))
