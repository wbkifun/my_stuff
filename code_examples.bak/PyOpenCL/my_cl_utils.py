#!/usr/bin/env python

import pyopencl as cl


def print_device_info(platforms, devices):
	print('Total Platforms : %d' % len(platforms))
	print('Total devices : %d' % len(devices))
	for i, platform in enumerate(platforms):
		print('\nPlatform #%d' % i)
		print('  name: %s' % platform.get_info(cl.platform_info.NAME))
		print('  version: %s' % platform.get_info(cl.platform_info.VERSION))

		for j, device in enumerate(devices):
			print('Device #%d' % j)
			print('  name: %s' % device.get_info(cl.device_info.NAME))
			print('  max compute unis: %d' % device.get_info(cl.device_info.MAX_COMPUTE_UNITS))
			print('  global mem size: %d' % device.get_info(cl.device_info.GLOBAL_MEM_SIZE))
			print('  local mem size: %d' % device.get_info(cl.device_info.LOCAL_MEM_SIZE))
			print('  constant mem size: %d' % device.get_info(cl.device_info.MAX_CONSTANT_BUFFER_SIZE))


def get_optimal_global_work_size(device):
	warp_size = 32
	max_resident_warp_dict = {
			'1.0':24, '1.1':24,
			'1.2':32, '1.3':32,
			'2.0':48}
	compute_capability = \
			str(device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV)) \
			+ '.' + str(device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV))
	max_resident_warp = max_resident_warp_dict[compute_capability]
	max_compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)

	return max_compute_units * max_resident_warp * warp_size


if __name__ == '__main__':
	devices = []
	queues = []
	platforms = cl.get_platforms()
	for platform in platforms:
		devices.extend(platform.get_devices())
	print_device_info(platforms, devices)

	print('Optimal Gs = %d' % get_optimal_global_work_size(devices[0]))
