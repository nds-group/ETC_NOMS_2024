The scripts in this folder are useful for extracting the data from the pcap files.
- run the _extract_pkts.sh_ script in the folder containing the downloaded pcap files to extract the packet features.
    - the packet files are saved in a _txt_files_ folder
- run the _extract_flows.sh_ script in the folder containing the downloaded pcap files to aggregate the packet data in the _txt_files_ folder into flow data saved in .csv files.
    - this bash script makes use of the _extract_flows_from_txt.py_ script which takes as input the txt file, the csv file which is the output, and the number of packets to consider in each flow.
- merge the generated flow files into a single csv