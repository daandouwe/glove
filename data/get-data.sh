#!/usr/bin/env bash

VERSION=8  # either enwik8 or enwik9
WIK=enwik${VERSION}
FIL=fil${VERSION}

if [ ! -f "wikifil.pl" ]; then
    wget https://raw.githubusercontent.com/facebookresearch/fastText/master/wikifil.pl
fi

if [ ! -f "${FIL}" ]; then
  # Get enwik.
  wget -c http://mattmahoney.net/dc/enwik${VERSION}.zip -P "${DATADIR}"
  unzip ${WIK}.zip
  rm ${WIK}.zip
  # Clean enwik.
  perl wikifil.pl ${WIK} > ${FIL}
fi
