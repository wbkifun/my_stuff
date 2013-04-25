#!/usr/bin/env python

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

glutInit("Hello, World")
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(400, 400)
glutCreateWindow("Hello, World")
glClearColor(0., 0., 0., 1.)
glutSetDisplayFuncCallback(display)
glutDisplayFunc()
glutMainLoop()

