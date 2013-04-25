#!/usr/bin/env python

import subprocess as sp
import time


class HostStatus:
	def __init__(s, hostname):
		s.hostname = hostname
		s.procs = []

		s.on_sshtest = False
		s.on_pingtest = False
		s.ssh_pass = None
		s.ping_pass = None

		s.mem_use = None
		s.mem_total = None
		s.cpu_names = []
		s.cpu_uses = []
		s.cpu_temps = []

		s.num_nvidia_gpus = None
		s.gpu_names = []
		s.gpu_uses = []
		s.gpu_temps = []
		s.gpu_fans = []
		s.gpu_gmus = []
		s.gmem_totals = []
		s.gmem_uses = []

		s.sshtest_cmd = 'ssh %s hostname' % hostname
		s.pingtest_cmd = 'ping -c 3 %s' % hostname
		s.set_cpu_names_cmd = 'ssh %s cat /proc/cpuinfo' % hostname
		s.set_nvidia_gpus_cmd = 'ssh %s lspci' % hostname
		s.set_gpu_names_cmd = 'ssh %s nvidia-smi -a' % hostname
		s.set_cpu_uses_cmd = 'ssh %s mpstat -P ALL' % hostname
		s.set_cpu_temps_cmd = 'ssh %s sensors' % hostname
		s.set_mem_use_cmd = 'ssh %s free -b' % hostname
		s.set_gpu_infos_cmd = 'ssh %s nvidia-smi -a' % hostname
		s.set_gmem_uses_cmd = 'ssh %s gfree' % hostname

		s.txts = []

	
	def proc_summit(s, cmd):
		s.procs.append( sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE) )


	def sshtest(s):
		stdout, stderr = s.procs.pop().communicate()
		errs = ['No route to host', 'Connection timed out']
		if len([err for err in errs if err in stderr]): s.ssh_pass = False
		else: 
			s.ssh_pass = True
			s.ping_pass = True


	def pingtest(s):
		stdout, stderr = s.procs.pop().communicate()
		if '3 received' in stdout: s.ping_pass = True
		else: s.ping_pass = False


	def connection_test(s):
		if s.procs[0].poll() != None: 
			if s.on_pingtest: 
				s.pingtest()
				s.on_pingtest = False
				if h.ping_pass: 
					h.proc_summit(h.sshtest_cmd)
					s.on_sshtest = True
				else: 
					h.proc_summit(h.pingtest_cmd)
					s.on_pingtest = True

			elif s.on_sshtest: 
				s.sshtest()
				s.on_sshtest = False
				if h.ssh_pass: 
					pass
				else:
					h.proc_summit(h.pingtest_cmd)
					s.on_pingtest = True


	def set_cpu_names(s):
		stdout, stderr = s.procs.pop().communicate()
		re_lst = ['(R)', '(TM)', 'CPU', '@']
		for line in stdout.splitlines():
			if 'model name' in line:
				st = line.split(': ')[1]
				for re in re_lst: st = st.replace(re,' ')
				s.cpu_names.append( ' '.join(st.split()) )
		for name in s.cpu_names:
			s.cpu_uses.append(0)
			s.cpu_temps.append(0)


	def set_num_nvidia_gpus(s):
		s.proc_summit(s.set_nvidia_gpus_cmd)
		stdout, stderr = s.procs.pop().communicate()
		lines = [line for line in stdout.lower().splitlines() if 'nvidia' in line]
		s.num_nvidia_gpus = len([line for line in lines if '3d controller' in line]) \
				+ len([line for line in lines if 'vga compatible controller' in line])


	def set_gpu_names(s):
		stdout, stderr = s.procs.pop().communicate()
		if 'command not found' not in stderr:
			s.gpu_names.extend( ['nVidia ' + line.split(': ')[1] for line in stdout.splitlines() if 'Product Name' in line] )
			for name in s.gpu_names:
				s.gpu_uses.append(0)
				s.gpu_temps.append(1)
				s.gpu_fans.append(0)
				s.gpu_gmus.append(0)
				s.gmem_totals.append(0)
				s.gmem_uses.append(0)


	def set_cpu_uses(s, proc):
		stdout, stderr = proc.communicate()
		for i, line in enumerate( stdout.splitlines()[4:] ):
			s.cpu_uses[i] = float( line.split()[2] )


	def set_mem_use(s, proc):
		stdout, stderr = proc.communicate()
		lines = stdout.splitlines()
		s.mem_total = int( lines[1].split()[1] )
		s.mem_use = int( lines[2].split()[2] )


	def set_cpu_temps(s, proc):
		stdout, stderr = proc.communicate()
		errs = ['command not found', 'No sensors found']
		if len([err for err in errs if err in stderr]): 
			s.cpu_temps[0] = 'N/A'
		else:
			idx = 0
			for line in stdout.splitlines():
				if 'Core ' in line:
					s.cpu_temps[idx] = float( line.split()[2].strip('+') )
					idx += 1


	def set_gpu_infos(s, proc):
		stdout, stderr = proc.communicate()
		uses, temps, fans, gmus = [], [], [], []
		for line in stdout.splitlines():
			if 'GPU\t\t\t:' in line:
				uses.append( float(line.split(': ')[1].rstrip('%')) )
			elif 'Temperature' in line:
				temps.append( float(line.split(': ')[1].rstrip('C')) )
			elif 'Fan Speed' in line:
				fans.append( float(line.split(': ')[1].rstrip('%')) )
			elif 'Memory' in line:
				gmus.append( float(line.split(': ')[1].rstrip('%')) )

		for i in range( len(s.gpu_names) ):
			s.gpu_uses[i] = uses[i]
			s.gpu_temps[i] = temps[i]
			s.gpu_fans[i] = fans[i]
			s.gpu_gmus[i] = gmus[i]

	
	def set_gmem_uses(s, proc):
		stdout, stderr = proc.communicate()
		for i, line in enumerate(stdout.splitlines()[1:]):
			s.gmem_totals[i] = int( line.split()[2] )
			s.gmem_uses[i] = int( line.split()[4] )


	def update_infos_start(s):
		s.proc_summit(s.set_cpu_uses_cmd)
		s.proc_summit(s.set_mem_use_cmd)
		s.proc_summit(s.set_cpu_temps_cmd)
		if len(s.gpu_names): 
			s.proc_summit(s.set_gpu_infos_cmd)
			s.proc_summit(s.set_gmem_uses_cmd)


	def update_infos_end(s):
		s.set_cpu_uses(s.procs.pop(0))
		s.set_mem_use(s.procs.pop(0))
		s.set_cpu_temps(s.procs.pop(0))
		if len(s.gpu_names): 
			s.set_gpu_infos(s.procs.pop(0))
			s.set_gmem_uses(s.procs.pop(0))



class DCGMon:
	def __init__(s):
		s.hosts = s.get_hosts()


	def get_hosts(s):
		import os
		fpath = os.path.expanduser('~/.dcgstat_hosts')
		hnames = open(fpath, 'r').read().split()
		disables = [hname for hname in hnames if hname.startswith('#')]
		for disable in disables: hnames.remove(disable)

		return [HostStatus(hname) for hname in hnames]


	def procs_terminate(s):
		print('Terminating subprocesses...')
		for h in s.hosts:
			for proc in h.procs:
				if proc.poll() == None:
					proc.terminate()
			print('%s,' % h.hostname), 
		print('Done')


	def byte_unit(s, byte):
		sizes = [1024**4, 1024**3, 1024**2, 1024]
		units = ['TB', 'GB', 'MB', 'KB']
		widths = [100, 10]

		for size, unit in zip(sizes, units):
			if byte >= size:
				byte = float(byte) / size
				ut = unit
				break

		if byte >= 100: return ' %d %s' % (int(byte), ut)
		elif byte >= 10: return '%2.1f %s' % (byte, ut)
		else: return '%1.2f %s' % (byte, ut)

	
	def format_mem_use(s, total, use):
		try:
			pctg = (float)(use) / total * 100
			return '%5d %% (%s/ %s)' % (pctg, s.byte_unit(use), s.byte_unit(total))
		except ZeroDivisionError:
			return 'N/A'


	def set_plain_texts(s, h, bodys=[], head=None):
		h.txts = ['%s :' % h.hostname]

		if head != None:
			h.txts.append(head)

		elif not h.ping_pass:
			h.txts.append('PING test failed')

		elif not h.ssh_pass:
			h.txts.append('SSH connection failed')

		elif len(bodys):
			h.txts.append('\n')
			h.txts.extend(bodys)

		else:
			txt = '\n' + ' '*22 + 'Cores' + ' '*14 + 'Temp    Mem'
			txt += '\n    CPU  :'
			nc = len(h.cpu_uses)
			if nc == 4: 
				txt += ''.join(['%5d %%' % cuse for cuse in h.cpu_uses])
			elif nc < 4: 
				txt += ''.join(['%5d %%' % cuse for cuse in h.cpu_uses]) + ' '*7*(4 - nc)
			elif nc > 4: 
				txt += ''.join(['%5d %%' % cuse for cuse in h.cpu_uses[:4]])

			if h.cpu_temps[0] == 'N/A': cpu_temp = 'N/A'
			else: cpu_temp = max(h.cpu_temps)
			txt += '%5d C' % cpu_temp
			txt += s.format_mem_use(h.mem_total, h.mem_use)

			if nc > 4: 
				sls = range(4, 4*(nc/4 + 1), 4)
				if nc%4 > 0: sls.append(None)
				line2 = ''
				for i0, i1 in zip(sls[:-1], sls[1:]):
					txt += '\n' + ' '*7 + ''.join(['%5d %%' % cuse for cuse in h.cpu_uses[slice(i0, i1)]])

			if len(h.gpu_names): 
				txt += '\n\n' + ' '*13 + 'Core' + ' '*11 + 'GMU    Fan   Temp    Mem'
				for i, (guse, gmu, gfan, gtemp, gmem_total, gmem_use) in \
						enumerate(zip(h.gpu_uses, h.gpu_gmus, h.gpu_fans, h.gpu_temps, h.gmem_totals, h.gmem_uses)):
					txt += '\n    GPU %d:%5d %%%12d %%%5d %%%5d C' % (i, guse, gmu, gfan, gtemp)
					txt += s.format_mem_use(gmem_total, gmem_use)

			h.txts.append(txt)


	def plain_test(s):
		for step in range(20):
			print('step : %d' % step)
			if step == 0:
				for h in s.hosts: 
					h.proc_summit(h.sshtest_cmd)
					s.set_plain_texts(h, head='SSH connecting...')
				time.sleep(1)
				for h in s.hosts: 
					if h.procs[0].poll() != None: 
						h.sshtest()
						if h.ssh_pass:
							s.set_plain_texts(h, head='SSH connected.')
							h.proc_summit(h.set_cpu_names_cmd)
						else:
							s.set_plain_texts(h, head='Ping testing...')
							h.proc_summit(h.pingtest_cmd)
							h.on_pingtest = True

			elif step == 1:
				for h in s.hosts:
					if h.ssh_pass:
						h.set_cpu_names()
						s.set_plain_texts(h, ['    CPU  : %s' % h.cpu_names[0]])
						h.proc_summit(h.set_gpu_names_cmd)
					else: h.connection_test()

			elif step == 2:
				for h in s.hosts:
					if h.ssh_pass:
						s.set_plain_texts(h, ['    CPU  : %s' % h.cpu_names[0]])
						h.set_gpu_names()
						if len(h.gpu_names):
							s.set_plain_texts(h, ['\n    GPU %d: %s' % (i, gname) for i, gname in enumerate(h.gpu_names)])
						else:
							h.set_num_nvidia_gpus()
							if h.num_nvidia_gpus:
								s.set_plain_texts(h, ['\n    GPU  : %d NVIDIA GPUs' % h.num_nvidia_gpus])
								if not h.nvidia_smi_exist:
									s.set_plain_texts(h, ['\n           nvidia-smi command not found.' % h.num_nvidia_gpus])
							else:
								s.set_plain_texts(h, ['\n    GPU  : No NVIDIA GPUs'])
						h.update_infos_start()
					else: h.connection_test()

			else:
				for h in s.hosts:
					if h.ssh_pass:
						h.update_infos_end()
						h.update_infos_start()
					else: h.connection_test()
					s.set_plain_texts(h)

			for h in s.hosts: 
				print(' '.join(h.txts))
				print('')
			print('-'*47)
			time.sleep(2)

		s.procs_terminate()


if __name__ == '__main__':
	DCGMon().plain_test()

# TODO
# test urwid listbox scrolling (more hosts)
# show information for non-GPU system
# show label toggle - foot frame
# show processor name toggle - replace contents
# show refresh time (set refresh time ?)

