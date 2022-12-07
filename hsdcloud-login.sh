#!/bin/bash

# This script logs into an HSDCloud instance and forwards a port so
# that a Jupyter notebook running remotely can be accessed locally.

# Create this file on the HSD dashboard and put it in your home directory
PEM_FILE="${HOME}/.nasa-laptop.pem"

# Your Jupyter notebook should be running on this port
RPORT=8888
# This port should be free on your local machine
LPORT=9998

# This is the name of the instance. It changes every time you start
# and stop your instance and is a command line arg to this script
INSTANCE_NAME=$1

# SSH in with your PEM file and forward the port
ssh -i "${PEM_FILE}" \
    -L ${LPORT}:localhost:${RPORT} \
    ubuntu@${INSTANCE_NAME}.compute-1.amazonaws.com
