# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:42:44 2018

@author: LZC
"""
import pandas as pd
import numpy as np
Data_1 = pd.DataFrame({'id':list(range(1,11)),
                       'name':['A1','A2','A3','A4','A5','A6','A7','A8','A9','A0'],
                       'num':['12','34','14','52','45','21','12','64','64','93'],
                       'level':['A','B','A','A','B','B','B','B','B','B'],
                       "city":['Beijing ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai', 'BEIJING ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai']
        })
print('Data1列名：',Data_1.columns)
#print('Data1:',Data_1.values,'\n')
print('Data1:\n',Data_1,'\n')

Data_2 = pd.DataFrame({'id':list(range(11,21)),
                       'name':['A1','A2','A3','A4','A5','A6','A7','A8','A9','A0'],
                       'num':['12','34','14','52','45','21','12','64','64','93'],
                       'level':['A','B','A','A','B','B','B','B','B','B'],
                       "age":[23,44,54,32,34,32,44,54,32,34]},
                        columns = ['id','name','num','level','age'])
print('Data2列名：',Data_1.columns)
print('Data2:\n',Data_2,'\n')

Data_3 = pd.merge(Data_1,Data_2,how = 'outer')
print(Data_3)
