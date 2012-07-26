#!/usr/bin/env python

import urwid

palette = [
	('head', 'white', 'black'),
	('foot', 'white', 'black'),
	('reveal focus', 'black', 'dark cyan', 'standout'),
	]

class Test:
	def __init__(s):
		s.num = 0
		s.content = urwid.SimpleListWalker([
			urwid.AttrMap(w, None, 'reveal focus') for w in [
				urwid.Text("This is a text string that is fairly long"),
				urwid.Divider("-"),
				urwid.Text("Short one"),
				urwid.Text("Another"),
				urwid.Divider("-"),
				urwid.Text("What could be after this?"),
				urwid.Divider("-"),
				urwid.Text("The end."),]
			])


	def show_head_input(s, input, raw):
		s.show_key.set_text("Pressed: " + " ".join([unicode(i) for i in input]))
		return input


	def show_foot_num(s, loop=None, user_data=None):
		s.num += 1
		s.show_num.set_text("num = %d" % s.num)
		loop.set_alarm_in(2, s.show_foot_num)


	def update_num(s):
		s.num += 1


	def exit_on_q(s, input):
		if input in ('q', 'Q'):
			raise urwid.ExitMainLoop()


	def main(s):
		s.listbox = urwid.ListBox(s.content) 
		s.show_key = urwid.Text("", wrap='clip')
		s.show_num = urwid.Text("")
		s.header = urwid.AttrMap(s.show_key, 'head')
		s.footer = urwid.AttrMap(s.show_num, 'foot')
		s.frame = urwid.Frame(s.listbox, s.header, s.footer)

		loop = urwid.MainLoop(s.frame, palette, input_filter=s.show_head_input, unhandled_input=s.exit_on_q)
		s.show_foot_num(loop)
		loop.run()


if __name__ == '__main__':
	Test().main()
