import logging
from sub import func1, func2


def func3():
    logging.info('(func3) Start')
    logging.info('(func3) Do something')
    logging.debug('(func3) Detail info')
    logging.warning('(func3) Caution')
    logging.info('(func3) End')


def main():
    logging.basicConfig( \
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=getattr(logging, 'DEBUG'))

    logging.info('[MAIN] Start')
    logging.info('Do something')
    func1()
    func2()
    func3()
    logging.debug('Detail info')
    logging.warning('Caution')
    logging.info('[MAIN] End')


if __name__ == '__main__':
    main()
