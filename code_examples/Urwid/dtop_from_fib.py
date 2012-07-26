#!/usr/bin/env python

import urwid
        
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
	palette = [
		('foot','light gray', 'black'),
		('key','light cyan', 'black'),
		('title', 'white', 'black',),
		('head', 'white', 'black'),
		]

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
