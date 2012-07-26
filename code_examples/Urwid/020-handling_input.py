#!/usr/bin/env python

import urwid

txt = urwid.Text('Hello World')
fill = urwid.Filler(txt, 'top')

def show_or_exit(input):
	if input in ('q', 'Q'):
		raise urwid.ExitMainLoop()
	txt.set_text(repr(input))


loop = urwid.MainLoop(fill, unhandled_input=show_or_exit)
loop.run()
