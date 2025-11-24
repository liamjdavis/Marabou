#!/bin/bash
curdir=$pwd
mydir="${0%/*}"

cd $mydir
git clone https://github.com/arminbiere/cadical.git
cd cadical
git fetch --all
git checkout 14593f819cb242082b127396218bd60889409737
sed -i 's/\r$//' configure
find scripts -type f -name "*.sh" -exec sed -i 's/\r$//' {} +
sed -i 's/\r$//' VERSION
sed -i 's/\r$//' LICENSE
sed -i 's/\r$//' makefile.in
chmod +x configure scripts/*.sh
./configure -a -shared
make

cd $curdir
