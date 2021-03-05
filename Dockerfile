FROM tsutomu7/scientific-python:mkl
USER root
COPY . /code
WORKDIR /code
RUN pip install --upgrade pip
RUN pip install -e .[parallel]
RUN mkdir /afs
