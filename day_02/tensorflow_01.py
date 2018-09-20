# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 11:12:39 2018

First one for tensorflow,do everything with a simple start.

@author: LZC
"""

from __future__ import print_function
import tensorflow
c = tensorflow.constant('F*ck U,world!')


# 两种输出方式

with tensorflow.Session() as Sess:
    print(c.eval()) 


sess = tensorflow.Session()
print(sess.run(c))
sess.close()