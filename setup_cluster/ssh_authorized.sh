#!/bin/bash
# This script will create an SSH key for each existing user and create
# an authorized_keys file with their public key.

# Directory containing user home directories
homeDirs=/home

for x in `ls $homeDirs`; do
    echo Creating SSH key for $x...

    if [[ -e $homeDirs/$x/.ssh/id_rsa.pub ]]; then
        echo "$x already has a public key"
    else
        su $x -c "ssh-keygen -N \"\""  
    fi

    cat $homeDirs/$x/.ssh/id_rsa.pub >> $homeDirs/$x/.ssh/authorized_keys
    chown $x:$x $homeDirs/$x/.ssh/authorized_keys
    chmod 600 $homeDirs/$x/.ssh/authorized_keys

done
