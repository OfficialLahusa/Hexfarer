#!/bin/sh
# The run registry's former name: forwards everything to chronicle.sh
exec "$(cd "$(dirname "$0")" && pwd)/chronicle.sh" "$@"
