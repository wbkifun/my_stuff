#!/usr/bin/env python

import urwid

palette = [('I say', 'default,bold', 'default', 'bold'),]

def new_question():
	return urwid.Edit(('I say', 'What is your name?\n'))

def new_answer(name):
	return urwid.Text(('I say', 'Nice to meet you, ' + name + repr(content) + '\n'))

content = urwid.SimpleListWalker([new_question()])
listbox = urwid.ListBox(content)

def update_on_cr(input):
	if input != 'enter': return

	focus_widget, position = listbox.get_focus()
	if not hasattr(focus_widget, 'edit_text'): return
	if not focus_widget.edit_text: raise urwid.ExitMainLoop()

	content[position+1:position+2] = [new_answer(focus_widget.edit_text)]

	if not content[position+2:position+3]: content.append(new_question())
	listbox.set_focus(position+2)

loop = urwid.MainLoop(listbox, palette, unhandled_input=update_on_cr)
loop.run()
