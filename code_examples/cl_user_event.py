import pyopencl as cl

platforms = cl.get_platforms()
devices = platforms[0].get_devices()
context = cl.Context(devices)

print context
evt = cl.UserEvent(context)
print evt
print evt.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)

print cl.command_execution_status.COMPLETE
evt.set_status(cl.command_execution_status.COMPLETE)
print evt
