#!/bin/bash

# Transcriptions
wget http://www.linguistics.ucsb.edu/sites/secure.lsit.ucsb.edu.ling.d7/files/sitefiles/research/SBC/SBCorpus.zip
unzip SBCorpus.zip
rm SBCorpus.zip

# Metadata
mkdir -p meta && cd meta
wget http://www.linguistics.ucsb.edu/sites/secure.lsit.ucsb.edu.ling.d7/files/sitefiles/research/SBC/metadata.zip
unzip metadata.zip
rm metadata.zip
cd ..

# Audio
mkdir -p wav
for (( i=1; i<=60; i++)); do
  if [ $i -le 10 ]; then
    wget "http://www.linguistics.ucsb.edu/corpusmedia/SBC00${i}.wav" wav/
  else
    wget "http://www.linguistics.ucsb.edu/corpusmedia/SBC0${i}.wav" wav/
  fi
done
