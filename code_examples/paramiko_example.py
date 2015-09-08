from __future__ import division
import paramiko
import sys
from glob import glob



#------------------------------------------------------------------------------
# date argument
#------------------------------------------------------------------------------
YYYYMMDDHH = sys.argv[1]
assert len(YYYYMMDDHH) == 10, 'wrong date format: %s'%YYYYMMDDHH
YYYYMMDD = YYYYMMDDHH[:8]
YYYYMM   = YYYYMMDDHH[:6]
YYYY     = YYYYMMDDHH[:4]
MM       = YYYYMMDDHH[4:6]
DDHH     = YYYYMMDDHH[6:]
HH       = YYYYMMDDHH[8:]
#------------------------------------------------------------------------------


#============================================================
# setup
#============================================================
KIM_version = '0.25'
host = '210.125.45.30'
uid = 'kiaps-sop'
pwd = 'kiaps-sop123'
local_src   = '/home/kiaps-sop/cylc/Src'
local_plot  = '/home/kiaps-sop/cylc/Src/Post/figdir/KIM/VERF/PLOT'
local_input = '/data/kiaps/kim-cm/KIM/inputdata/ne120/vgecore'
local_base  = '/scratch/kiaps/kiaps-sop'
remote_base = '/data/kiaps-sop'


# the point at which is generated the DATE directory
dst_src_link = { \
        'KIM/COLD/LOG/'       : '%s/KIM/exp_ne120np4/cold_start/%s/exp.log'%(local_src,YYYYMMDDHH), \
        'KIM/COLD/MODL/FCST/' : '%s/%s/cold/%s/*.nc'%(local_base,KIM_version,YYYYMMDDHH), \
        'KIM/COLD/MODL/ANCIL/': ['%s/sst/%s*'%(local_input,YYYYMMDD), \
                                 '%s/seaice/%s*'%(local_input,YYYYMMDD)], \
        'KIM/COLD/MODL/INIT/' : '%s/atm/%s_%s*'%(YYYYMMDD,HH), \
        'KIM/COLD/VERF/PLOT/' : ['%s/COLD/NE120/%s/%s/%s/sfc/sfc.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/COLD/NE120/%s/%s/%s/850/850.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/COLD/NE120/%s/%s/%s/700/700.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/COLD/NE120/%s/%s/%s/500/500.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/COLD/NE120/%s/%s/%s/300/300.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/COLD/NE120/%s/%s/%s/n50/n50.gif'%(local_plot,YYYY,MM,DDHH)], \
        'KIM/WARM/LOG/'       : '/home/kiaps-sop/cylc/Src/KIM/exp_ne120np4/warm_start/YYYYMMDDHH/exp.log', \
        'KIM/WARM/MODL/FCST/' : '%s/%s/warm/%s/*.nc'%(local_base,KIM_version,YYYYMMDDHH), \
        'KIM/WARM/MODL/INIT/' : '/data/kiaps/kim-cm/KIM/inputdata/ne120/vgecore/atm/KIMInit/*%s*'%(YYYYMMDDHH), \
        'KIM/WARM/ANAL/KPOP/' : '%s/%s/warm/%s/KPOP_OUT/*'%(local_base,KIM_version,YYYYMMDDHH), \
        'KIM/WARM/ANAL/HYDA/' : ['%s/%s/warm/%s/HYBDA_OUT/*'%(local_base,KIM_version,YYYYMMDDHH), \
                                 '%s/%s/warm/%s/HYBDA_IN/*'%(local_base,KIM_version,YYYYMMDDHH), \
                                 '%s/%s/warm/%s/AnalInc/*'%(local_base,KIM_version,YYYYMMDDHH)], \
        'KIM/WARM/VERF/PLOT/' : ['%s/WARM/NE120/%s/%s/%s/sfc/sfc.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/WARM/NE120/%s/%s/%s/850/850.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/WARM/NE120/%s/%s/%s/700/700.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/WARM/NE120/%s/%s/%s/500/500.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/WARM/NE120/%s/%s/%s/300/300.gif'%(local_plot,YYYY,MM,DDHH), \
                                 '%s/WARM/NE120/%s/%s/%s/n50/n50.gif'%(local_plot,YYYY,MM,DDHH)], \
        'UM/MODL/FCST/'       : None, \
        'UM/VERF/PLOT/'       : None}
#============================================================



'''
#------------------------------------------------------------------------------
# remote command
#------------------------------------------------------------------------------
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(host, username=uid, password=pwd)
stdin, stdout, stderr = ssh.exec_command('uptime')
ssh.close()

print 'stdin', stdin
print 'stdout', stdout.readline()
print 'stderr', stderr.readlines()
'''


#------------------------------------------------------------------------------
# transport files
#------------------------------------------------------------------------------
transport = paramiko.Transport((host, 22))
transport.connect(username=uid, password=pwd)
sftp = paramiko.SFTPClient.from_transport(transport)

for dst, src in dst_src_link.items():
    #----------------------------------------------------
    # make the YYYYMM/DDHH directory
    #----------------------------------------------------
    sftp.chdir(remote_base + dst)
    try:
        sftp.chdir(YYYYMM)
    except IOError:
        sftp.mkdir(YYYYMM) 
        sftp.chdir(YYYYMM)

    sftp.mkdir(DDHH) 
    sftp.chdir(DDHH)


    #----------------------------------------------------
    # copy target files
    #----------------------------------------------------
    fpaths = list()

    if src == None:
        pass
    elif type(src) == list:
        for s in src:
            fpaths.extend( glob(s) )
    else:
        fpaths.extend( glob(src) )

    for fpath in fpaths:
        sftp.put(fpath, '.')

sftp.close()
