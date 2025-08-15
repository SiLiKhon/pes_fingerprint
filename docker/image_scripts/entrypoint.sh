#!/bin/bash

source /app/pfp_env/bin/activate
cd /workdir
exec "$@"
