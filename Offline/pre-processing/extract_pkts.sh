for f in *.pcap
	do
		echo $f
        tshark -r $f -Y 'ip.proto == 6 or ip.proto == 17' -T fields -e frame.time_relative -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e ip.proto -e ip.len -e udp.srcport -e udp.dstport -E separator='|' > ./txt_files/$f.txt
	done
