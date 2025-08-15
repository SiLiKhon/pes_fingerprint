#!/bin/bash

DOCKER_BUILD_PATH=`realpath "$(dirname ${BASH_SOURCE[0]})"`
TMP_SRC="build_tmp/source"

if [ "$(pwd)" != "$DOCKER_BUILD_PATH" ]; then
    echo "Changing directory to docker build root: $DOCKER_BUILD_PATH"
    cd "$DOCKER_BUILD_PATH"
fi

if [ -d $TMP_SRC ]; then
    rm -r $TMP_SRC
fi

mkdir -p $TMP_SRC
cd ".." && \
  cp --parent $(git ls-files .) "$DOCKER_BUILD_PATH/$TMP_SRC" && \
  cd -

docker build -t pes_fingerprint:latest --progress=plain .
