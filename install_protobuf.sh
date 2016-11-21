#!/bin/sh
if [ ! -d "$HOME/protobuf/lib" ]; then
    wget https://github.com/google/protobuf/archive/v3.1.0.tar.gz
    tar -xzvf v3.1.0.tar.gz
    pushd protobuf-3.1.0/ && ./autogen.sh && ./configure --prefix=$HOME/protobuf && make && sudo make install && popd
    pushd protobuf-3.1.0/python && python setup.py build && sudo python setup.py install && popd
else
    echo "Using cached protobuf directory."
fi
