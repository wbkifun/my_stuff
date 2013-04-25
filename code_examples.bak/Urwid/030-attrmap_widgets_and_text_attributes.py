#!/usr/bin/env python

import urwid

palette = [('banner', 'black', 'light gray', 'standout,underline'),
		('streak', 'black', 'dark red', 'standout'),
		('bg', 'black', 'dark blue'),]

txt = urwid.Text(('banner', "Hello World"), align='center')
map1 = urwid.AttrMap(txt, 'streak')
fill = urwid.Filler(map1)
map2 = urwid.AttrMap(fill, 'bg')

def exit_on_q(input):
	if input in ('q', 'Q'):
		raise urwid.ExitMainLoop()

loop = urwid.MainLoop(map2, palette, unhandled_input=exit_on_q)
loop.run()
