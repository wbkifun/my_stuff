from time import sleep, localtime, strftime, gmtime
from datetime import datetime, timedelta




def print_progress(t0, step, max_step):
    '''
    Assume step starts from 0
    '''
    step += 1
    dt = datetime.now() - t0
    step_width = max(len(str(max_step)), 4)

    if step == 1:
        print('[{:>{ms}}] [progress] [elapsed time] [total elapsed time]'.format(
            'step', ms=step_width))

    print('[{:>{ms}d}] [{:>6.1f} %] [{:>12s}] [{:>18s}]'.format(
            step, 
            (step)/max_step*100, 
            str(timedelta( seconds=int(dt.seconds) )), 
            str(timedelta( seconds=int(max_step*dt.seconds/step) )),
            ms=step_width),
            flush=True, end='\r')



if __name__ == '__main__':
    tmax = 120
    t0 = datetime.now()

    for tstep in range(tmax):
        print_progress(t0, tstep, tmax)
        sleep(0.7)

    print()
