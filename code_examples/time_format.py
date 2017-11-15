import time, datetime


t1 = '2015-07-31T18:57:04Z'
t2 = '2015-08-01T02:34:00Z'

st1 = time.strptime(t1, '%Y-%m-%dT%H:%M:%SZ')   # Convert to the time struct
st2 = time.strptime(t2, '%Y-%m-%dT%H:%M:%SZ')

dt = time.mktime(st2) - time.mktime(st1)        # Convert to seconds
dt_str = str(datetime.timedelta(seconds=dt))    # Convert to the format as HH:MM:SS

print(dt_str)


#
# Add a timedelta using time.strptime
#
t3 = '2013070100'
st3 = time.strptime(t3, '%Y%m%d%H')
dt = time.mktime(st3) + 30*3600     # 30 hours (seconds)
sdt = time.localtime(dt)
print(time.strftime("%Y%m%d%H", sdt))



#
# Add a timedelta using datetime.datetime.strptime
#
st4 = '2015070100'
t4 = datetime.datetime.strptime(st4, '%Y%m%d%H')
print('t0     = {}'.format(t4))
for hour in [-3, -2, -1, 0, 1, 2, 3]:
    t5 = t4 + datetime.timedelta(hours=hour)
    st5 = t5.strftime('%Y%m%d%H')
    sign = '-' if hour < 0 else '+'
    print('t0 {} {} = {}'.format(sign, abs(hour), st5))
