#!/usr/bin/env python

import curses
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()		# without requiring the Enter key
stdscr.keypad(1)

pad = curses.newpad(100, 100)
for y in range(100):
	for x in range(100):
		try: pad.addch(y, x, ord('a') + (x * x + y * y) % 26)
		except curses.error: pass
pad.refresh(0, 0, 5, 5, 20, 75)

stdscr.keypad(0)
curses.nocbreak()
curses.echo()
curses.endwin()
