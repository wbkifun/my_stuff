#!/bin/sh

ioc64 -device='cpu' -cmd='compile' -bo='-Ibuild/' -input=apb_ext.cl -ir=build/apb_ext.ir
ioc64 -device='cpu' -cmd='compile' -bo='-Ibuild/' -input=apb.cl -ir=build/apb.ir
ioc64 -device='cpu' -cmd='link' -binary='build/apb_ext.ir,build/apb.ir' -ir=build/apb.clbin

