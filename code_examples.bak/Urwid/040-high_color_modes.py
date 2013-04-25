#!/usr/bin/env python

import urwid

palette = [
		('banner', '', '', '', '#ffa', '#60d'),
		('streak', '', '', '', 'g50', '#60a'),
		('inside', '', '', '', 'g38', '#808'),
		('outside', '', '', '', 'g27', '#a06'),
		('bg', '', '', '', 'g7', '#d06'),]

txt = urwid.Text(('banner', "Hello World"), align='center')
map1 = urwid.AttrMap(txt, 'streak')
pile = urwid.Pile([
	urwid.AttrMap(urwid.Divider(), 'outside'),
	urwid.AttrMap(urwid.Divider(), 'inside'),
	map1,
	urwid.AttrMap(urwid.Divider(), 'inside'),
	urwid.AttrMap(urwid.Divider(), 'outside')])
fill = urwid.Filler(pile)
map2 = urwid.AttrMap(fill, 'bg')

def show_or_exit(input):
	if input in ('q', 'Q'):
		raise urwid.ExitMainLoop()

loop = urwid.MainLoop(map2, palette, unhandled_input=show_or_exit)
loop.run()
