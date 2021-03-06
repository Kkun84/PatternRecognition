FROM python:3.8

# for Timezone setting
RUN apt-get update && apt-get install -y --no-install-recommends tzdata

# SSH setting
# https://docs.docker.com/engine/examples/running_ssh_service/#build-an-eg_sshd-image
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:pass' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthetication/PasswordAuthetication/' /etc/ssh/sshd_config
# # SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Install Python library
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Install
RUN apt-get update && apt-get install -y --no-install-recommends \
    fish nano git sudo curl

# for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxrender1

ENV PYTHONPATH ${PYTONPATH}:"./"

RUN chmod 777 /root
ENV HOME /root

CMD ["/bin/bash"]
