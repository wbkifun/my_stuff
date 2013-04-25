#!/usr/bin/env python

import curses, traceback


def file_func():
	s = curses.newwin(5,10,2,1)
	s.box()
	s.addstr(1,2, "I", hotkey_attr)
	s.addstr(1,3, "nput", menu_attr)
	s.addstr(2,2, "O", hotkey_attr)
	s.addstr(2,3, "utput", menu_attr)
	s.addstr(3,2, "T", hotkey_attr)
	s.addstr(3,3, "ype", menu_attr)
	s.addstr(1,2, "", hotkey_attr)
	s.refresh()
	c = s.getch()
	if c in (ord('I'), ord('i'), curses.KEY_ENTER, 10):
		curses.echo()
		s.erase()
		screen.addstr(5,33, " "*43, curses.A_UNDERLINE)
		cfg_dict['source'] = screen.getstr(5,33)
		curses.noecho()
	else:
		curses.beep()
		s.erase()

	return CONTINUE


def doit_func():
	pass

def help_func():
	pass


def main(stdscr):
	# Frame the interface area at fixed VT100 size
	global screen
	screen = stdscr.subwin(23, 79, 0, 0)
	screen.box()
	screen.hline(2, 1, curses.ACS_HLINE, 77)
	screen.refresh()

	# Define the topbar menus
	file_menu = ("File", "file_func()")
	doit_menu = ("Do It!", "doit_func()")
	help_menu = ("Help", "help_func()")
	exit_menu = ("Exit", "EXIT")

	# Add the topbar menus to screen object
	topbar_menu((file_menu, doit_menu, help_menu, exit_menu)) 

	# Enter the topbar menu loop
	while topbar_key_handler():
		draw_dict()


if __name__=='__main__':
	try:
		# Initialize curses
		stdscr=curses.initscr()
		# Turn off echoing of keys, and enter cbreak mode,
		# where no buffering is performed on keyboard input
		curses.noecho()
		curses.cbreak()

		# In keypad mode, escape sequences for special keys
		# (like the cursor keys) will be interpreted and
		# a special value like curses.KEY_LEFT will be returned
		stdscr.keypad(1)
		main(stdscr)                    # Enter the main loop

		# Set everything back to normal
		stdscr.keypad(0)
		curses.echo()
		curses.nocbreak()
		curses.endwin()                 # Terminate curses

	except:
		# In event of error, restore terminal to sane state.
		stdscr.keypad(0)
		curses.echo()
		curses.nocbreak()
		curses.endwin()
		traceback.print_exc()           # Print the exception

