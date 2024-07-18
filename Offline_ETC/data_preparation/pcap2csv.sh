#!/bin/bash
pcapfile=$1

tshark -o gui.column.format:"SP,%uS,DP,%uD" -r "${pcapfile}" -T fields -E header=y -E separator=, -e frame.number -e frame.time_epoch -e frame.time_delta -e ip.src -e _ws.col.SP -e ip.dst -e _ws.col.DP -e ip.proto -e frame.len   
