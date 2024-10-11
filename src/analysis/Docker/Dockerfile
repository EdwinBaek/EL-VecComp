# Cuckoo Sandbox Dockerfile for PE File Analysis
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Switch to root user
USER root

# Add MongoDB repository
RUN apt-get update && apt-get install -y wget gnupg && \
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add - && \
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    python2.7 python-dev libffi-dev libssl-dev libcap2-bin curl libxml2-dev mongodb-org postgresql \
    libxslt1-dev libjpeg-dev zlib1g-dev libpq-dev tcpdump apparmor-utils yara libfuzzy-dev nano \
    build-essential libffi-dev gcc g++ make automake autoconf libtool pkg-config wget \
    qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst virt-manager \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ssdeep from source
RUN wget https://github.com/ssdeep-project/ssdeep/releases/download/release-2.14.1/ssdeep-2.14.1.tar.gz && \
    tar -xzvf ssdeep-2.14.1.tar.gz && \
    cd ssdeep-2.14.1 && \
    ./configure && make && make install && \
    cd .. && rm -rf ssdeep-2.14.1 ssdeep-2.14.1.tar.gz

# Update shared library cache
RUN ldconfig

# Install pip for Python 2
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py && python2.7 get-pip.py && rm get-pip.py

RUN pip2 install --no-cache-dir pip==20.3.4 setuptools==44.1.1 wheel ssdeep==3.4 cuckoo==2.0.7 celery==4.4.7 \
    redis==3.5.3 requests==2.27.1 pefile==2019.4.18 python-dotenv==0.18.0 watchdog==0.10.3

RUN setcap cap_net_raw,cap_net_admin=eip /usr/sbin/tcpdump

RUN useradd -m -s /bin/bash cuckoo

# 상위 폴더의 dataset을 참조하도록 변경
RUN mkdir -p /app/dataset && chown -R cuckoo:cuckoo /app/dataset
RUN mkdir -p /home/cuckoo/.cuckoo && chown -R cuckoo:cuckoo /home/cuckoo/.cuckoo

# Set permissions
RUN chown -R cuckoo:cuckoo /home/cuckoo

USER cuckoo
WORKDIR /home/cuckoo
RUN cuckoo init

VOLUME ["/app/dataset"]

CMD ["/bin/bash"]