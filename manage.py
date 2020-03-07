# -*- coding: utf-8 -*-
# @Time    : 20-2-7 下午2:52
# @Author  : zhuzhengyi

from flask_script import Manager
from App import create_app

app = create_app()
manager = Manager(app=app)

'''
python3 manage.py runserver
'''
if __name__ == '__main__':
	manager.run()