#!/usr/bin/env python

from __future__ import division
from subprocess import Popen, PIPE, STDOUT



#---------------------------------------------
# setup
#---------------------------------------------
pic_format = 'png'
fps = 25                    # movie speed

'''
mpeg4 : Best quality, higher compression and more options, 
        but on Windows requires the installation of the DivX codec
        (from www.divx.com)

msmpeg4v2 : The Microsoft MPEG4 V2 codec. 
            On average the files are 10% bigger, 
            but nothing has to be installed on Windows.
'''
codec = 'msmpeg4v2'



#---------------------------------------------
# size
#---------------------------------------------
ps = Popen(['ls'], stdout=PIPE)
stdout = ps.communicate()
fnames = stdout[0].split()
for fname in fnames:
    if fname.endswith(pic_format): break

ps = Popen(['identify', fname], stdout=PIPE)
stdout = ps.communicate()
size = stdout[0].split()[2]
width, height = [int(x) for x in size.split('x')]



#---------------------------------------------
# optimal bitrate
#---------------------------------------------
# The 50 factor can vary between 40 and 60 to trade quality for size
optimal_bitrate = 50 * 25 * width * height // 256



#---------------------------------------------
# options
#---------------------------------------------
if codec == 'mpeg4':
    opt = "vbitrate=%d:mbd=2:keyint=132:v4mv:vqmin=3:lumi_mask=0.07:dark_mask=0.2:mpeg_quant:scplx_mask=0.1:tcplx_mask=0.1:naq" % optimal_bitrate

elif codec == 'msmpeg4v2':
    opt = "vbitrate=%d:mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" % optimal_bitrate



#---------------------------------------------
# two pass compression for better compression and quality
#---------------------------------------------
cmd1 = 'mencoder -ovc lavc -lavcopts vcodec=%s:vpass=1:%s -mf type=%s:fps=%d -nosound -o /dev/null mf://*.%s' % (codec, opt, pic_format, fps, pic_format)

cmd2 = 'mencoder -ovc lavc -lavcopts vcodec=%s:vpass=2:%s -mf type=%s:fps=%d -nosound -o output.avi mf://*.%s' % (codec, opt, pic_format, fps, pic_format)


for cmd in [cmd1, cmd2]:
    print cmd
    ps = Popen(cmd.split())
    ps.wait()
