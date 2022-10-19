from celery import Celery
from celery.signals import worker_process_init


cel_app = Celery('tasks', broker='redis://localhost/1', backend='redis://localhost/2')

import time


@worker_process_init.connect()
def on_worker_init(**_):
    print('----- INIT done !')




@cel_app.task(name='txt2img')
def txt2img(params=None):
    time.sleep(2)
    print(params)
    return 'OK'


@cel_app.task(name='inpaint')
def inpaint(params=None):
    time.sleep(4)
    print(params)
    return 'OK'
