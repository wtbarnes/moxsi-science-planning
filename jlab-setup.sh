#!/bin/bash

start_jlab_server () {
    screen -dm -S "jlab" bash -c "conda activate moxsipipeline; jupyter lab --port=8888 --no-browser --ip=*"
}


# Send a "Ctrl-C" to a detached screen session

stop_jlab_server () {
	screen -S "jlab" -X stuff "^C"
}

# Kill a detached screen session

kill_jlab_server () {
	screen -X -S "jlab" quit
}
