for f in ./txt_files/*.txt
	do
		echo $f
        python3 extract_flows_from_txt.py $f $f.csv 8
	done
