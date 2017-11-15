import logging
import subprocess as sp



def func1():
    logging.info('(func1) Start')
    logging.info('(func1) Do something')
    logging.debug('(func1) Detail info')
    logging.warning('(func1) Caution')
    logging.info('(func1) End')



def func2():
    logging.info('(func2) Start')
    logging.info('(func2) Call external executable')

    proc = sp.Popen('ls -l ./', stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    stdout, stderr = proc.communicate()
    logging.info('----- Shell execute %s', '-'*30)
    logging.info('----- standard output -----\n%s, %s', stdout.decode(), 'aaa')
    if len(stderr) > 0:
        logging.warning('----- standard errorr -----\n%s', stderr.decode())
    logging.info('-'*50)

    logging.debug('(func2) Detail info')
    logging.warning('(func2) Caution')
    logging.info('(func2) End')
