#!/usr/bin/env python

import curses, traceback


def main(stdscr):
	MESSAGE_X = 0
	MESSAGE_Y = 5

	def send_xmms_arg(char):
		os.system('ssh ' + user + '@' + host + ' "xmms -' + char + '"')

	def write_instructions():
		stdscr.addstr(0, 0, "c: play/pause; b: next; z: prev")
		stdscr.refresh()

	def write_message(message):
		stdscr.addstr(MESSAGE_Y, MESSAGE_X, message) # clear out 
		stdscr.refresh()

	def do_play_pause():
		#send_xmms_arg('u')
		write_message("XMMS was paused / unpaused")

	def do_next():
		#send_xmms_arg('f')
		write_message("Skipped to the next song.")

	def do_prev():
		#send_xmms_arg('r')
		write_message("Skipped to the previous song.")

	write_instructions()

	while 1:
		c = stdscr.getch()
		if c == ord('q'): break
		elif c == ord('c'): do_play_pause()
		elif c == ord('b'): do_next()
		elif c == ord('z'): do_prev()


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

