#! /bin/sh
### BEGIN INIT INFO
# Provides:          var_subdirs
# Required-Start:    checkroot
# Required-Stop:
# Default-Start:     S
# Default-Stop:
# Short-Description: Make the temporary /var subdirectories
### END INIT INFO


. /lib/lsb/init-functions

case "$1" in
    start|"")
        log_begin_msg "Make the subdirectories in /var."
        /etc/init.d/var_subdirs.py
        log_end_msg $?
        ;;
    restart|reload|force-reload)
        echo "Error: argument '$1' not supported" >&2
        exit 3
        ;;
    stop)
        ;;
    *)
        echo "Usage: $0 start|stop" >&2
        exit 3
        ;;
esac

exit 0
