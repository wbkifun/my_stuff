#!/usr/bin/env python

import os, sys
import subprocess as sp
import threading
import urwid as uw
import time


class HostStatus:
	def __init__(s):
		s.hosts = s._get_hosts()
		s.infos = dict((host, {}) for host in s.hosts)


	def _get_hosts(s):
		fpath = os.path.expanduser('~/.dmcgstat_hosts')
		hosts = open(fpath, 'r').read().split()
		disableds = []
		for host in hosts:
			if host.startswith('#'):
				disableds.append(host)
		for disabled in disableds:
			hosts.remove(disabled)

		return hosts


	def _get_cpu_names(s, host):
		cmd = 'ssh %s cat /proc/cpuinfo' % (host)
		stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
		lines = stdout.splitlines()

		re_lst = ['(R)', '(TM)', 'CPU', '@']
		cpu_names = []
		for line in lines:
			if 'model name' in line:
				st = line.split(': ')[1]
				for re in re_lst: st = st.replace(re,' ')
				cpu_names.append( ' '.join(st.split()) )

		return cpu_names


	def _get_gpu_names(s, host):
		cmd = 'ssh %s nvidia-smi -a' % (host)
		stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
		lines = stdout.splitlines()

		gpu_names = []
		for line in lines:
			if 'Product Name' in line:
				gpu_names.append( line.split(': ')[1] )

		return gpu_names


	def _pingtest(s, host):
		cmd = 'ping -c 3 -W 2 %s' % (host)
		ps = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
		stdout, stderr = ps.communicate()
		if '3 received' in stdout:
			s.infos[host]['ping'] = True
			s._sshtest(host)
		else:
			s.infos[host]['ping'] = False


	def _sshtest(s, host):
		cmd = 'ssh -o ConnectTimeout=2 %s hostname' % (host)
		stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
		errs = ['No route to host', 'Connection timed out']
		if len([err for err in errs if err in stderr]):
			s.infos[host]['ssh'] = False
			s._pingtest(host)
		else:
			s.infos[host]['ssh'] = True
			s.infos[host]['ping'] = True


	def _get_mem_info(s, host):
		cmd = 'ssh %s free -m' % (host)
		stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
		lines = stdout.splitlines()

		total = int( lines[1].split()[1] )
		used = int( lines[2].split()[2] )

		return float(used)/total*100, used, total


	def _get_cpu_info(s, host):
		cmd = 'ssh %s mpstat -P ALL' % (host)
		stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
		lines = stdout.splitlines()

		uses = []
		for line in lines[4:]:
			uses.append( float(line.split()[2]) )

		return uses
	

	def _get_gpu_info(s, host):
		cmd = 'ssh %s nvidia-smi -a' % (host)
		stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
		lines = stdout.splitlines()

		uses = []
		temps = []
		fans = []
		mems = []
		for line in lines:
			if 'GPU\t\t\t:' in line:
				uses.append( float(line.split(': ')[1].rstrip('%')) )
			elif 'Temperature' in line:
				temps.append( float(line.split(': ')[1].rstrip('C')) )
			elif 'Fan Speed' in line:
				fans.append( float(line.split(': ')[1].rstrip('%')) )
			elif 'Memory' in line:
				mems.append( float(line.split(': ')[1].rstrip('%')) )

		return uses, temps, fans, mems
	

	def _set_cpu_gpu_names(s, host):
		s._sshtest(host)
		if s.infos[host]['ssh']:
			s.infos[host]['cpu_names'] = s._get_cpu_names(host)
			s.infos[host]['gpu_names'] = s._get_gpu_names(host)
		else:
			s._pingtest(host)


	def _update_infos(s, host):
		if s.infos[host]['ssh']:
			# memory
			r1, r2, r3 = s._get_mem_info(host)
			s.infos[host]['mem_use'] = r1
			s.infos[host]['mem_use_mib'] = r2
			s.infos[host]['total_mem_mib'] = r3

			# cpu
			r1 = s._get_cpu_info(host)
			s.infos[host]['cpu_uses'] = r1

			# gpu
			r1, r2, r3, r4 = s._get_gpu_info(host)
			s.infos[host]['gpu_uses'] = r1
			s.infos[host]['gpu_temps'] = r2
			s.infos[host]['gpu_fans'] = r3
			s.infos[host]['gpu_mems'] = r4

		else:
			s._sshtest(host)



class HostStatusView(HostStatus):
	def __init__(s):
		HostStatus.__init__(s)
		s.palette = s._get_palette()
		s.update_interval = 1	# seconds
		s.iter_num = 0

		s.threads = dict((host, None) for host in s.hosts)
		s.uw_texts = dict((host, uw.Text([('hostname', host), ' : ', 'SSH connecting...'], wrap='clip')) for host in s.hosts)

	
	def _get_palette(s):
		return [
				('head',	'dark cyan',	'black'),
				('foot',	'light gray',	'black'),
				('key',		'light cyan',	'black'),
				('hostname',	'white',	'black'),
				]


	def _get_frame(s, listbox):
		header_lst = [
				('head', "Status Viewer for Distributed MEM/CPU/GPU Utilizations\n"),
				]
		footer_lst = [
				('key', "UP"), ", ", 
				('key', "DOWN"), ", ",
				('key', "PAGE UP"), ", ", 
				('key', "PAGE DOWN"), " move   ",
				('key', "Q"), " exits",
				]

		s.footer_txt = uw.Text(footer_lst, wrap='clip')

		footer = uw.AttrMap(s.footer_txt, 'foot')
		header = uw.AttrMap(uw.Text(header_lst, wrap='clip'), 'head')
		return uw.Frame(listbox, header=header, footer=footer)


	def _set_uw_text(s, host):
		info = s.infos[host]

		txts = [('hostname', host), ' : ']
		if not info['ping']:
			txts.append('PING test failed')
		elif not info['ssh']:
			txts.append('SSH connection failed')
		else:
			# memory
			r1 = info['mem_use']
			r2 = info['mem_use_mib']
			r3 = info['total_mem_mib']
			txts.append('\n    MEM   :%5d %%  (%d / %d MiB)' % (r1, r2, r3))

			# cpu
			r1 = info['cpu_uses']
			r2 = info['cpu_names'][0]
			txts.append('\n    CPU(%d):' % len(r1))
			for cpu_use in r1:
				txts.append('%5d %%' % cpu_use)
			txts.append('   (%s)' % r2)

			# gpu
			r1 = info['gpu_uses']
			r2 = info['gpu_temps']
			r3 = info['gpu_fans']
			r4 = info['gpu_mems']
			r5 = info['gpu_names']
			txts.append('\n\n              Core   Temp    Fan    GMU')
			for i, (guse, gtemp, gfan, gmem, gname) in enumerate( zip(r1, r2, r3, r4, r5) ):
				txts.append('\n    GPU %d :%5d %%%5d C%5d %%%5d %%   (%s)' % (i, guse, gtemp, gfan, gmem, gname))

		s.uw_texts[host].set_text(txts)


	def _update_content(s, loop=None, user_data=None):
		if s.iter_num == 0:
			s.iter_num = 1
			s.mainloop.set_alarm_in(0, s._update_content)

		elif s.iter_num == 1:
			for host in s.hosts:
				s.threads[host] = threading.Thread(target=s._set_cpu_gpu_names, args=(host,))
				s.threads[host].start()
			for host in s.hosts:
				s.threads[host].join()
				thread = threading.Thread(target=s._update_infos, args=(host,))
				thread.start()
			s.iter_num = 2
			s.mainloop.set_alarm_in(0, s._update_content)

		else:
			for host in s.hosts:
				info = s.infos[host]
				thread = s.threads[host]

				if info['ssh'] == False and thread.is_alive():
					pass
				else:
					thread.join()
					s._set_uw_text(host)
					thread = threading.Thread(target=s._update_infos, args=(host,))
					thread.start()

			s.mainloop.set_alarm_in(s.update_interval, s._update_content)
			

	def exit_on_q(s, input):
		if input in ('q', 'Q'):
			raise uw.ExitMainLoop()


	def main(s):
		walk_list = []
		for host in s.hosts:
			walk_list.append( s.uw_texts[host] )
			walk_list.append( uw.Divider(' ') )

		content = uw.SimpleListWalker(walk_list)
		frame = s._get_frame( uw.ListBox(content) )
		s.mainloop = uw.MainLoop(frame, s.palette, unhandled_input=s.exit_on_q)
		s._update_content()
		s.mainloop.run()
		print("'q' pressed. Terminating threads..."),
		for thread in s.threads.values(): thread.join()
		print('done')



if __name__ == '__main__':
	s = HostStatusView()
	'''
	for host in s.hosts:
		s._sshtest(host)
		print host, s.infos[host]['ssh']
	'''
	s.main()



