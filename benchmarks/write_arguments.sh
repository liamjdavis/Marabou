#!/usr/bin/env bash

for version in old new
do
    for exp in DeepCert NAP AltLoops
    do
        for file in $(ls "$exp"/)
        do
            echo $exp $version $(basename $file)
        done
    done
    for exp in VeriX
    do
        while read i
        do
            echo $exp $version $i
        done < VeriX/correctly_classified_index.txt
    done
done

