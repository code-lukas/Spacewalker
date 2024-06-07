#!/usr/bin/env sh

# apt packages
sudo apt-get update && sudo apt-get install -y libgl1

# pip packages
pip install -r ./requirements.txt

# npm installations
npm install -g npm@latest
npm i

