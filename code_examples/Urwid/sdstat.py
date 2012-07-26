#!/usr/bin/env python

import urwid as uw
import subprocess as sp

hostname
ssh_connection

mem_rsc
mem_rsc_mib
total_mem_mib

num_cpu_core
cpu_rscs
cpu_name

gpu_rscs
gpu_temps
gpu_fans
gpu_mems
gpu_names



class HostsStatus:
	if __init__(s):
		s.nodes = []


	def get_host_info(s):
		import os
		fpath = os.path.expanduser('~/.sdstat_hosts')
		hosts = open(fpath, 'r').read().split()
		pass


	def update_resources(s):
		cmd = [
		sp.Popen
		pass


	def get_host_head_info(s, node_index):
		node = s.nodes[node_index]
		e1 = node['hostname']

		txt = '%s :' % e1
		if not s.ssh_connection:
			txt += ' SSH Connection Error!'

		return txt


	def get_mem_rsc_str(s, node_index):
		node = s.nodes[node_index]
		e1 = node['mem_rsc']
		e2 = node['mem_rsc_mib']
		e3 = node['total_mem_mib']

		return '\t%3d %%  (%d / %d MiB)' % (e1, e2, e3)


	def get_cpu_rsc_str(s, node_index):
		node = s.nodes[node_index]
		e1 = node['num_cpu_core']
		e2s = node['cpu_rscs']
		e3 = node['cpu_name']

		txt = '\tCPU(%d)' % e1
		for crsc in e2s:
			txt += '\t%3d %%' % crsc
		txt += '\t(%s)' % e3

		return txt


	def get_gpu_rsc_str(s, node_index):
		node = s.nodes[node_index]
		e1s = node['gpu_rscs']
		e2s = node['gpu_temps']
		e3s = node['gpu_fans']
		e4s = node['gpu_mems']
		e5s = node['gpu_names']

		txt = '\n\t\t\tCore\tTemp\tFan \tGMU\n'
		for i, (grsc, gtemp, gfan, gmem, gname) in enumerate( zip(e1s, e2s, e3s, e4s, e5s) ):
			txt += '\tGPU #%d\t%3d\t%3d\t%3d\t%3d\t%s\n' % (i, grsc, gtemp, gfan, gmem, gname)

		return txt



palette = [
	('title', 'white', 'black',),
	('head', 'white', 'black'),
	('foot', 'light gray', 'black'),
	('key', 'light cyan', 'black'),
	('hostname', 'white,standout', 'black'),
	]


def host_connect_status(host):
	return uw.Text([('hostname', host['hostname']), host['connect_status']])


def host_utilizations(host):
	wigets = [uw.Text('\tMEM\t\t%3d %% (%d/%d MiB)' % (host['mem_rsc'], host['mem_rsc_mib'], host['total_mem_mib']))]
	for i, (cpu_rsc, cpu_temp, cpu_fan) in enumerate( zip(host['cpu_rsc'], host['cpu_temp'], host['cpu_fan']) ):

	
			uw.Text('\t%s\t\t%3d %%\t%3d C\t%3d %%' % (host['cpu_name'], host['cpu_rsc'], host['cpu_temp'], host['cpu_fan']))
			]



class ViewWalker(urwid.ListWalker):
	""" positions returned are (value at position-1, value at poistion) tuples. """
	def __init__(self):
		self.focus = (0L,1L)
		self.numeric_layout = NumericLayout()

	def _get_at_pos(self, pos):
		"""Return a widget and the position passed."""
		return urwid.Text("%d, %s, %s" % (len(pos), pos[0], pos[1]), layout=self.numeric_layout), pos

	def get_focus(self): 
		return self._get_at_pos(self.focus)

	def set_focus(self, focus):
		self.focus = focus
		self._modified()

	def get_next(self, start_from):
		a, b = start_from
		#focus = b, a+b
		focus = a, b+1
		return self._get_at_pos(focus)

	def get_prev(self, start_from):
		a, b = start_from
		#focus = b-a, a
		focus = a, b-1
		if focus[1] < 0: return None
		else: return self._get_at_pos(focus)


class NumericLayout(urwid.TextLayout):
	""" TextLayout class for bottom-right aligned numbers """
	def layout( self, text, width, align, wrap ):
		""" Return layout structure for right justified numbers. """
		lt = len(text)
		r = lt % width # remaining segment not full width wide
		print lt, r
		linestarts = range( r, lt, width )
		#lout = [[(width, x, x+width)] for x in linestarts]	# fill the rest of the lines
		lout = [[(width, 10, 10+width)] for x in linestarts]	# fill the rest of the lines
		if r:
			return [[(width-r,None),(r, 0, r)]] + lout	# right-align the remaining segment on 1st line
		else:
			return lout


def main():
	footer_text = [
		('title', "Distributed MEM/CPU/GPU Utilizations"), "    ",
		('key', "UP"), ", ", ('key', "DOWN"), ", ",
		('key', "PAGE UP"), " and ", ('key', "PAGE DOWN"),
		" move  ",
		('key', "Q"), " exits",
		]

	header_text = [
		('Hostname    Device        Utilization         Temperature    Fan Speed\n'),
		('--------------------------------------------------------------------------'),
		]

	def exit_on_q(input):
		if input in ('q', 'Q'):
			raise urwid.ExitMainLoop()

	listbox = urwid.ListBox(ViewWalker())
	header = urwid.AttrMap(urwid.Text(header_text), 'head')
	footer = urwid.AttrMap(urwid.Text(footer_text), 'foot')
	frame = urwid.Frame(listbox, header=header, footer=footer)
	loop = urwid.MainLoop(frame, palette, unhandled_input=exit_on_q)
	loop.run()


if __name__=="__main__": 
	main()
