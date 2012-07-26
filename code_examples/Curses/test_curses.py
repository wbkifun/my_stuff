#!/usr/bin/env python

import curses





def restorescreen():
	curses.nocbreak()
	curses.echo()
	curses.endwin()


def main():
	scr = curses.initscr()
	curses.noecho()
	curses.cbreak()

	scr.addstr(12,25, "Hello!")
	scr.refresh()

	while True:
		c = scr.getch()
		c = chr(c)
		scr.addstr(12,30, c)
		scr.refresh()
		
		if c == 'q':
			break
	restorescreen()


if __name__ == '__main__':
	try:
		main()
	except:
		restorescreen()
		import traceback
		traceback.print_exc()

