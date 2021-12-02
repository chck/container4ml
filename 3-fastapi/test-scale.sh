#!/bin/bash
#
# ref: https://gist.github.com/g105b/34ec96a305b74087d5a64db27d1b9fec
#
target=${1:-http://127.0.0.1:2222}
while true
do
  date
  # 100 req * 10000
  hey -n 10000 -c 100 $target
  sleep 3
done
