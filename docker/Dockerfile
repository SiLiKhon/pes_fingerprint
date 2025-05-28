FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt update && \
    apt install -y python3.10 python3.10-dev python3-pip python3-venv

RUN python3 -m venv /app/pfp_env
RUN . /app/pfp_env/bin/activate && \
    pip3 install --upgrade pip setuptools

RUN . /app/pfp_env/bin/activate && \
    python3 -m pip install \
        jupyter \
        m3gnet==0.2.4 \
        tensorflow==2.13 \
        pymatgen==2023.9.25

RUN mkdir -p /app/pes_fingerprint
COPY build_tmp/source /app/pes_fingerprint
RUN . /app/pfp_env/bin/activate && \
    pip3 install -e /app/pes_fingerprint

COPY image_scripts/entrypoint.sh /app
RUN mkdir /workdir && chmod -R 777 /workdir && chmod -R 777 /app

ENTRYPOINT [ "/app/entrypoint.sh", "pfp-run-ase" ]
