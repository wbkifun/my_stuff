#!/usr/bin/env python

import os
import sys
import subprocess as sp
import threading
import urwid
from datetime import datetime


class HostStatus:
	def __init__(s, hostname):
		s.hostname = hostname
		s.ssh_test = False
		s.ping_test = False

		s.mem_use = None
		s.mem_use_mib = None
		s.total_mem_mib = None

		s.cpu_name = None
		s.cpu_uses = []

		s.gpu_exist = False
		s.gpu_names = []
		s.gpu_uses = []
		s.gpu_temps = []
		s.gpu_fans = []
		s.gpu_mems = []

		s.subp = None
		s.thread = None


	def sshtest(s):
		if s.subp == None:
			cmd = 'ssh %s hostname' % (s.hostname)
			s.subp = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
			stdout, stderr = s.subp.communicate()
			s.subp = None

			errs = ['No route to host', 'Connection timed out']
			if len([err for err in errs if err in stderr]):
				s.ssh_test = False
			else:
				s.ssh_test = True
				s.ping_test = True


	def pingtest(s):
		if s.subp == None:
			cmd = 'ping -c 3 %s' % (s.hostname)
			s.subp = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
			stdout, stderr = s.subp.communicate()
			s.subp = None

			stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
			if '3 received' in stdout: s.ping_test = True
			else: s.ping_test = False


	def set_cpu_name(s):
		if s.subp == None:
			cmd = 'ssh %s cat /proc/cpuinfo' % (s.hostname)
			s.subp = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
			stdout, stderr = s.subp.communicate()
			s.subp = None

			stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
			lines = stdout.splitlines()

			re_lst = ['(R)', '(TM)', 'CPU', '@']
			for line in lines:
				if 'model name' in line:
					st = line.split(': ')[1]
					for re in re_lst: st = st.replace(re,' ')
					break
			s.cpu_name = ' '.join(st.split())


	def set_gpu_names(s):
		if s.subp == None:
			cmd = 'ssh %s lspci' % (s.hostname)
			s.subp = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
			stdout, stderr = s.subp.communicate()
			s.subp = None

			stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
			lines = [line for line in stdout.lower().splitlines() if 'nvidia' in line]
			num_gpu = len([line for line in lines if '3d controller' in line]) \
					+ len([line for line in lines if 'vga compatible controller' in line])
			if num_gpu > 0:
				s.gpu_exist = True
				cmd = 'ssh %s nvidia-smi -a' % (s.hostname)
				stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
				if 'command not found' in stderr:
					print('%s : nvidia-smi command not found' % s.hostname)
					sys.exit()
				else:
					s.gpu_names = [line.split(': ')[1] for line in stdout.splitlines() if 'Product Name' in line]


	def set_cpu_info(s):
		if s.subp == None:
			cmd = 'ssh %s mpstat -P ALL' % (s.hostname)
			s.subp = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
			stdout, stderr = s.subp.communicate()

			s.cpu_uses = [float(line.split()[2]) for line in stdout.splitlines()[4:]]

			# memory
			cmd = 'ssh %s free -m' % (s.hostname)
			s.subp = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
			stdout, stderr = s.subp.communicate()
			s.subp = None

			lines = stdout.splitlines()
			total = int( lines[1].split()[1] )
			used = int( lines[2].split()[2] )

			s.mem_use = float(used)/total*100
			s.mem_use_mib = used
			s.total_mem_mib = total


	def set_gpu_info(s):
		if s.subp == None:
			cmd = 'ssh %s nvidia-smi -a' % (s.hostname)
			s.subp = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
			stdout, stderr = s.subp.communicate()
			s.subp = None

			stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()

			uses, temps, fans, mems = [], [], [], []
			for line in stdout.splitlines():
				if 'GPU\t\t\t:' in line:
					uses.append( float(line.split(': ')[1].rstrip('%')) )
				elif 'Temperature' in line:
					temps.append( float(line.split(': ')[1].rstrip('C')) )
				elif 'Fan Speed' in line:
					fans.append( float(line.split(': ')[1].rstrip('%')) )
				elif 'Memory' in line:
					mems.append( float(line.split(': ')[1].rstrip('%')) )

			s.gpu_uses = uses
			s.gpu_temps = temps
			s.gpu_fans = fans
			s.gpu_mems = mems
	

	def update_names(s):
		s.set_cpu_name()
		s.set_gpu_names()


	def update_infos(s):
		s.set_cpu_info()
		if s.gpu_exist: s.set_gpu_info()



class HostsStatusView():
	def __init__(s):
		s.hosts = s.get_hosts()
		s.palette = s.get_palette()

		s.urwid_texts = dict((h.hostname, urwid.Text([('hname', h.hostname), ' : ', 'SSH connecting...'], wrap='clip')) for h in s.hosts)
	

	def get_hosts(s):
		fpath = os.path.expanduser('~/.dcgstat_hosts')
		names = open(fpath, 'r').read().split()
		disables = [name for name in names if name.startswith('#')]
		for disable in disables: names.remove(disable)

		return [HostStatus(name) for name in names]


	def get_palette(s):
		return [
				('head',	'dark cyan',	'black'),
				('foot',	'light gray',	'black'),
				('key',		'light cyan',	'black'),
				('hname',	'white',	'black'),
				]


	def get_frame(s, listbox):
		header_lst = [
				('head', "Status Viewer for Distributed MEM/CPU/GPU Utilizations\n"),
				]
		s.footer_lst = [
				"\n",
				('key', "UP"), ", ", 
				('key', "DOWN"), ", ",
				('key', "PAGE UP"), ", ", 
				('key', "PAGE DOWN"), " move   ",
				('key', "Q"), " exits",
				]

		s.footer_text = urwid.Text(s.footer_lst, wrap='clip')

		footer = urwid.AttrMap(s.footer_text, 'foot')
		header = urwid.AttrMap(urwid.Text(header_lst, wrap='clip'), 'head')
		return urwid.Frame(listbox, header=header, footer=footer)


	def set_urwid_text(s, h):
		txts = [('hostname', h.hostname), ' : ']
		if h.ping_test != None and not h.ping_test:
			txts.append('PING test failed')

		elif not h.ssh_test:
			txts.append('SSH connection failed')

		else:
			# memory
			txts.append( '\n    MEM  :%5d %%  (%d / %d MiB)' \
					% (h.mem_use, h.mem_use_mib, h.total_mem_mib) )

			# cpu
			tmp1 = '\n           '
			tmp2 = '\n    CPU  :'
			for i, cuse in enumerate(h.cpu_uses):
				tmp1 += ' Core%d' % i
				tmp2 += '%5d %%' % cuse

			txts.append(tmp1 + tmp2)

			# gpu
			if h.gpu_exist:
				txts.append('\n\n             Core   Temp    Fan    GMU')
				for i, (guse, gtemp, gfan, gmem) in \
						enumerate( zip(h.gpu_uses, h.gpu_temps, h.gpu_fans, h.gpu_mems) ):
					txts.append('\n    GPU %d:%5d %%%5d C%5d %%%5d %%' % (i, guse, gtemp, gfan, gmem))

		s.urwid_texts[h.hostname].set_text(txts)


	def show_refresh_time_foot(s, t0):
		dt0 = datetime.now() - t0
		dt = dt0.seconds + dt0.microseconds*1e-6
		txt_lst = ["\n", "refresh time : %1.2f s" % dt]
		txt_lst.extend(s.footer_lst)
		s.footer_text.set_text(txt_lst)


	def update_content(s, mainloop, user_data=None):
		t0 = datetime.now()

		if user_data == None:
			for h in [h for h in s.hosts if h.ssh_test]:
				h.thread = threading.Thread(target=h.update_infos)
				h.thread.start()

			'''
			for h in [h for h in s.hosts if not h.ssh_test]:
				if h.ping_test: func = h.sshtest
				else: func = h.pingtest
				if not h.thread.is_alive():
					h.thread.join()
					h.thread = threading.Thread(target=func)
					h.thread.start()
			'''

			for h in s.hosts:
				if h.ssh_test: h.thread.join()
				s.set_urwid_text(h)

			s.show_refresh_time_foot(t0)
			mainloop.set_alarm_in(2.0, s.update_content)

		elif user_data == 0: 
			for h in s.hosts: 
				h.thread = threading.Thread(target=h.sshtest)
				h.thread.start()

			mainloop.set_alarm_in(0, s.update_content, 1)

		elif user_data == 1: 
			ssh_tests = [h.ssh_test for h in s.hosts]
			while True not in ssh_tests:
				ssh_tests = [h.ssh_test for h in s.hosts]
			import time
			time.sleep(1)

			for h in [h for h in s.hosts if h.ssh_test]:
				h.thread.join()
				h.thread = threading.Thread(target=h.update_names)
				h.thread.start()

			for h in [h for h in s.hosts if h.ssh_test]:
				h.thread.join()
				h.thread = threading.Thread(target=h.update_infos)
				h.thread.start()

			for h in s.hosts:
				if h.ssh_test: h.thread.join()
				s.set_urwid_text(h)

			s.show_refresh_time_foot(t0)
			mainloop.set_alarm_in(0, s.update_content)

			

	def exit_on_q(s, input):
		if input in ('q', 'Q'): raise urwid.ExitMainLoop()


	def main(s):
		walk_list = []
		for h in s.hosts:
			walk_list.append( s.urwid_texts[h.hostname] )
			walk_list.append( urwid.Divider(' ') )

		content = urwid.SimpleListWalker(walk_list)
		frame = s.get_frame( urwid.ListBox(content) )
		mainloop = urwid.MainLoop(frame, s.palette, unhandled_input=s.exit_on_q)
		s.update_content(mainloop, 0)
		mainloop.run()
		print("'q' pressed. Terminating threads..."),
		for h in s.hosts: 
			if h.thread.is_alive:
				h.thread.join(1)
		print('done')



if __name__ == '__main__':
	s = HostsStatusView()
	'''
	h = s.hosts[2]
	print h.hostname
	h.sshtest()
	h.update_names()
	print h.gpu_exist
	h.update_infos()
	s.set_urwid_text(h)
	'''
	s.main()
