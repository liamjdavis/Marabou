#!/bin/bash
curdir=$pwd
mydir="${0%/*}"
version=$1
protoc_path=${2:-$(command -v protoc)}

onnxdir=onnx-$version

# This script downloads the protobuffer file from the ONNX repo.
# There are no C++ bindings for ONNX so we compile the protobuffer ourselves.
# see https://stackoverflow.com/questions/67301475/parse-an-onnx-model-using-c-extract-layers-input-and-output-shape-from-an-on
# for details.

if [ -z "$protoc_path" ]; then
    echo "Error: protoc not found. Pass the path as second argument or ensure it is on PATH."
    exit 1
fi

cd $mydir

mkdir -p $onnxdir
cd $onnxdir
echo "Downloading ONNX proto file"
wget -q https://raw.githubusercontent.com/onnx/onnx/v$version/onnx/onnx.proto3 -O onnx.proto3

echo "Compiling the ONNX proto file"
$protoc_path --cpp_out=. onnx.proto3

cd $curdir
