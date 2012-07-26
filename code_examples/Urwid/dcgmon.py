#!/usr/bin/env python

import subprocess as sp
import urwid


class HostStatus:
	def __init__(s, hostname):
		s.hostname = hostname
		s.cpu_name = ''
		s.ping_err = False
		s.rsh_err = False

		cmd0 = 'rsh %s ' % s.hostname
		s.cmds = {'ping_test': 'ping -c 3 %s' % s.hostname, 
				'rsh_test': cmd0 + 'hostname',
				'hostname': cmd0 + 'hostname',
				'cpu_name': cmd0 + 'cat /proc/cpuinfo',
				'cpu_temp': cmd0 + 'sensors',
				'cpu': cmd0 + 'mpstat -P ALL',
				'mem': cmd0 + 'free -b',
				'num_nvidia_gpu': cmd0 + 'lspci',
				'gpu': cmd0 + 'nvidia-smi -a',
				'gmem': cmd0 + 'gfree'}

		s.procs = {}
		s.txt = ''


	
	def submit_proc(s, job):
		cmd = s.cmds[job]
		return sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)


	def submit_jobs(s):
		jobs = ['hostname', 'cpu', 'cpu_temp', 'mem', 'gpu', 'gmem']
		for job in jobs: 
			s.procs[job] = s.submit_proc(job)


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
			return '    N/A'


	def set_cpu_name(s):
		if s.cpu_name == '':
			# Get the CPU name via 'cat /proc/cpuinfo'
			names = []
			re_lst = ['(R)', '(TM)', 'CPU', '@']
			proc = s.submit_proc('cpu_name')
			stdout, stderr = proc.communicate()
			for line in stdout.splitlines():
				if 'model name' in line:
					st = line.split(': ')[1]
					for re in re_lst: st = st.replace(re,' ')
					names.append( ' '.join(st.split()) )

			s.cpu_name = names[0]


	'''
	def get_num_nvidia_gpu(s):
		# Get the number of NVIDIA GPUs via 'lspci'
		proc = s.submit_proc('num_nvidia_gpu')
		stdout, stderr = proc.communicate()
		lines = [line for line in stdout.lower().splitlines() if 'nvidia' in line]
		num_gpus = len([line for line in lines if '3d controller' in line]) \
				+ len([line for line in lines if 'vga compatible controller' in line])

		return num_gpus
	'''


	def update_txt_cpu(s, print_opt='normal'):
		# Get the CPU usage via 'mpstat -P ALL'
		uses = []
		stdout, stderr = s.procs['cpu'].communicate()
		for i, line in enumerate( stdout.splitlines()[4:] ):
			uses.append( float(line.split()[2]) )

		# Get the CPU temperature via 'sensors'
		temp = 0
		stdout, stderr = s.procs['cpu_temp'].communicate()
		for line in stdout.splitlines():
			if 'Core ' in line:
				temp2 = float( line.split()[2].strip('+') )
				if temp2 > temp: temp = temp2

		# Get the CPU memory usage via 'free -b'
		stdout, stderr = s.procs['mem'].communicate()
		lines = stdout.splitlines()
		mem_total = int( lines[1].split()[1] )
		mem_use = int( lines[2].split()[2] )

		# Return print string
		if print_opt == 'device_name':
			s.set_cpu_name()
			txt = '\n    CPU  : %s' % s.cpu_name
		else:
			if print_opt == 'normal':
				txt = '\n' + ' '*22 + 'Cores' + ' '*14 + 'Temp    Mem'
			elif print_opt == 'compact':
				txt = ''

			nc = len(uses)
			txt += '\n    CPU  :'
			if nc == 4: 
				txt += ''.join(['%5d %%' % use for use in uses])
			elif nc < 4: 
				txt += ''.join(['%5d %%' % use for use in uses]) + ' '*7*(4 - nc)
			elif nc > 4: 
				txt += ''.join(['%5d %%' % use for use in uses[:4]])

			if temp == 0: txt += '%7s' % ('N/A')
			else: txt += '%5d C' % temp
			txt += s.format_mem_use(mem_total, mem_use)

			if nc > 4: 
				sls = range(4, 4*(nc/4 + 1), 4)
				if nc%4 > 0: sls.append(None)
				for i0, i1 in zip(sls[:-1], sls[1:]):
					txt += '\n' + ' '*7 + ''.join(['%5d %%' % use for use in uses[slice(i0, i1)]])

		s.txt = txt


	def update_txt_gpu(s, print_opt='normal'):
		# Get the GPU informations via 'nvidia-smi -a'
		names, uses, temps, fans, gmus = [], [], [], [], []
		stdout, stderr = s.procs['gpu'].communicate()
		lines = stdout.splitlines()
		for line in stdout.splitlines():
			if 'Product Name' in line:
				names.append( 'Nvidia ' + line.split(': ')[1] )
			elif 'GPU\t\t\t:' in line:
				uses.append( float(line.split(': ')[1].rstrip('%')) )
			elif 'Temperature' in line:
				temps.append( float(line.split(': ')[1].rstrip('C')) )
			elif 'Fan Speed' in line:
				fans.append( float(line.split(': ')[1].rstrip('%')) )
			elif 'Memory' in line:
				gmus.append( float(line.split(': ')[1].rstrip('%')) )


		# Get the GPU memory usage via 'gfree'
		mem_totals, mem_uses = [], []
		stdout, stderr = s.procs['gmem'].communicate()
		for i, line in enumerate(stdout.splitlines()[1:]):
			mem_totals.append( int(line.split()[2]) )
			mem_uses.append( int(line.split()[4]) )


		# Return print string
		if len(names) > 0:
			if print_opt == 'device_name':
				for i, name in enumerate(names):
					txt = '\n    GPU %d: %s' % (i, name)

			else:
				if print_opt == 'normal':
					txt = '\n\n' + ' '*13 + 'Core' + ' '*11 + 'GMU    Fan   Temp    Mem'
				elif print_opt == 'compact':
					txt = ''

				for i, (use, gmu, fan, temp, mem_total, mem_use) in \
						enumerate(zip(uses, gmus, fans, temps, mem_totals, mem_uses)):
						txt += '\n    GPU %d:%5d %%%12d %%%5d %%%5d C' % (i, use, gmu, fan, temp)
						txt += s.format_mem_use(mem_total, mem_use)
			s.txt += txt


	def get_txt(s):
		#print s.hostname
		if s.ping_err:
			print 'ping'
			s.txt = 'PING test failed'

		elif s.rsh_err:
			print 'rsh'
			if s.procs.has_key('ping_test'):
				stdout, stderr = s.procs.pop('ping_test').communicate()
				if '3 received' in stdout:
					s.ping_err = False
					s.txt = 'RSH connection failed.'
				else: 
					s.ping_err = True
					s.txt = 'PING test failed.'
			else:
				s.txt = 'RSH connection failed'

		elif not s.procs['hostname'].poll() == None:
			#print 'proc'
			stdout, stderr = s.procs['hostname'].communicate()
			if stderr == '': 
				#print 'update_cpu'
				s.txt = ''
				s.update_txt_cpu()
				#print 'update_gpu'
				s.update_txt_gpu()
				#print 'submit_job'
				s.submit_jobs()
			elif 'rsh: Could not make a connection.' in stderr:
				print 'rsh_err'
				s.rsh_err = True
				s.procs['ping_test'] = s.submit_proc('ping_test')

		return s.txt



class DCGMon:
	def __init__(s):
		s.hosts = s.get_hosts()
		for host in s.hosts: 
			host.submit_jobs()
			host.utxt = urwid.Text([('hname', host.hostname), ' : ', 'RSH connecting...'], wrap='clip')

		s.palette = [
				('head',	'dark cyan',	'black'),
				('foot',	'light gray',	'black'),
				('key',		'light cyan',	'black'),
				('hname',	'white',	'black'), ]


	def get_hosts(s):
		import os
		fpath = os.path.expanduser('~/.dcgstat_hosts')
		hostnames = open(fpath, 'r').read().splitlines()

		return [HostStatus(hostname) for hostname in hostnames if not hostname.startswith('#')]


	def get_frame(s, listbox):
		header_lst = [
				('head', "Status Viewer for Distributed MEM/CPU/GPU Utilizations\n"), ]
		footer_lst = [
				"\n",
				('key', "UP"), ", ", 
				('key', "DOWN"), ", ",
				('key', "PAGE UP"), ", ", 
				('key', "PAGE DOWN"), " move   ",
				('key', "Q"), " exits", ]

		footer_text = urwid.Text(footer_lst, wrap='clip')
		footer = urwid.AttrMap(footer_text, 'foot')
		header = urwid.AttrMap(urwid.Text(header_lst, wrap='clip'), 'head')

		return urwid.Frame(listbox, header=header, footer=footer)


	def update_content(s, loop=None, user_data=None):
		for host in s.hosts: 
			utxt_list = [('hname', host.hostname), ' : ']
			utxt_list.append( host.get_txt() )
			host.utxt.set_text(utxt_list)
		loop.set_alarm_in(2, s.update_content)


	def terminate_procs(s):
		print('Terminating subprocesses...')
		for host in s.hosts:
			print('%s,' % host.hostname), 
			for proc in host.procs.values():
				if proc.poll() == None:
					proc.terminate()
		print('Done')


	def exit_on_q(s, input):
		if input in ('q', 'Q'): raise urwid.ExitMainLoop()


	def main(s):
		walk_list = []
		for host in s.hosts:
			walk_list.append( host.utxt )
			walk_list.append( urwid.Divider(' ') )

		content = urwid.SimpleListWalker(walk_list)
		frame = s.get_frame( urwid.ListBox(content) )
		loop = urwid.MainLoop(frame, s.palette, unhandled_input=s.exit_on_q)
		s.update_content(loop)
		loop.run()
		s.terminate_procs()



if __name__ == '__main__':
	DCGMon().main()


# TODO
# test urwid listbox scrolling (more hosts)
# show information for non-GPU system
# show label toggle - foot frame
# show processor name toggle - replace contents
# show refresh time (set refresh time ?)

