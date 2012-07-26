#!/usr/bin/env python

import pyopencl as cl

ctx = cl.create_some_context()

evt = cl.Event()
'''
evt = cl.UserEvent(ctx)
print dir(evt)

status = cl.command_execution_status.QUEUED
print status
evt.set_status(status)
evt.wait()
'''
