import pandas as pd
import numpy as np
import sys

filename_in  = sys.argv[1]
filename_out = sys.argv[2]
npkts        = int(sys.argv[3])

packet_data = pd.DataFrame()

packet_data = pd.read_csv(filename_in, sep = '|', header=None)

packet_data.columns = ['timestamp', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'ip.proto', 'ip.len','udp.srcport', 'udp.dstport']

packet_data = packet_data[(packet_data["ip.proto"] != "1,17") & (packet_data["ip.proto"] != "1,6")].reset_index(drop=True)
packet_data = packet_data.dropna(subset=['ip.proto'])
packet_data["ip.src"] = packet_data["ip.src"].astype(str)
packet_data["ip.dst"] = packet_data["ip.dst"].astype(str)
packet_data["ip.len"] = packet_data["ip.len"].astype("int")
##
packet_data["tcp.srcport"] = packet_data["tcp.srcport"]
packet_data["tcp.dstport"] = packet_data["tcp.dstport"]
packet_data["udp.srcport"] = packet_data["udp.srcport"].astype('Int64')
packet_data["udp.dstport"] = packet_data["udp.dstport"].astype('Int64')
#
packet_data["srcport"] = np.where(packet_data["ip.proto"] == "6", packet_data["tcp.srcport"], packet_data["udp.srcport"])
packet_data["dstport"] = np.where(packet_data["ip.proto"] == "6", packet_data["tcp.dstport"], packet_data["udp.dstport"])
#
packet_data["srcport"] = np.where(packet_data["ip.proto"] == 6, packet_data["tcp.srcport"], packet_data["udp.srcport"])
packet_data["dstport"] = np.where(packet_data["ip.proto"] == 6, packet_data["tcp.dstport"], packet_data["udp.dstport"])
#
packet_data["srcport"] = packet_data["srcport"].astype('Int64')
packet_data["dstport"] = packet_data["dstport"].astype('Int64')

#===============================CREATE THE FLOW IDs AND DROP UNWANTED COLUMNS =============================================#
packet_data = packet_data.drop(["tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"],axis=1)
packet_data = packet_data.reset_index(drop=True)

packet_data["flow.id"] = packet_data["ip.src"].astype(str) + " " + packet_data["ip.dst"].astype(str) + " " + packet_data["srcport"].astype(str) + " " + packet_data["dstport"].astype(str) + " " + packet_data["ip.proto"].astype(str)


# Labeling
filename_patterns = {"background"      : "Background", 
                     "webbrowsing"     : "WebBrowsing",
                     "youtube"         : "YouTube",
                     "gmail"           : "Gmail",
                     "discord"         : "Discord",
                     "whatsapp"        : "WhatsApp",
                     "signal"          : "Signal",
                     "telegram"        : "Telegram",
                     "messenger"       : "Messenger",
                     "teams"           : "Teams"
                    }

for pattern, labeld in filename_patterns.items():
    if pattern in filename_in:
        label = labeld

number_of_pkts_limit, min_number_of_packets = npkts, npkts
#===============================Extract flows from packets and calculate features=============================================#
main_packet_size = {}  # dictionary to store list of packet sizes for each flow (Here key = flowID, value = list of packet sizes)
flow_list = []  # contains the flowIDs (a combination of SIP,DIP,srcPort, dstPort, proto)
main_inter_arrival_time = {}  # dictionary to store list of IATs for each flow (Here key = flowID, value = list of IATs)
last_time = {}  # for each flow we store timestamp of the last packet arrival

avg_pkt_sizes = {}  # contains the flowID and their calculated average packet sizes
string = {}  # For each flow, we have a string of feature values (just for printing purpose, on screen)

labels = {}  # contains the flowID and their labels
packet_count = {}  # contains flowID as key and number of packets as valu

# ==============================================================================================================================#
print("NOW: COLLECTING PACKETS INTO FLOWS...")
for row in packet_data.itertuples(index=True, name='Pandas'):
    time    = float(row[1])    # timestamp of the packet
    srcip   = row[2]          #src ip
    dstip   = row[3]          #dst ip
    pktsize = row[5]        #packet size   
    proto   = row[4]         #protocol
    srcport = row[6]     #source port
    dstport = row[7]     #destination port
    key     = row[8]          #key which is a concatenation of the 5-tuple to identify the flow

    if key in flow_list:  # check if the packet belongs to already existing flow ?
        if (len(main_packet_size[key]) < number_of_pkts_limit ):
            packet_count[key] = packet_count[key] + 1  # increment packet count
            main_packet_size[key].append(pktsize)  # append its packet size to the packet size list for this flow
            lasttime = last_time[key]
            diff = round(float(time) - float(lasttime), 9)  # calculate inter-arrival time (seconds)
            main_inter_arrival_time[key].append(diff)  # append IAT
            ##
            labels[key] = label
            ##
            last_time[key] = time  # update last time for the flow, to the timestamp of this packet


    else:  # if this packet is the first one in this NEW flow
        flow_list.append(key)  # make its entry in the existing flow List
        packet_count[key] = 1  # first packet arrived for this flow, set count =1
        main_packet_size[key] = [pktsize]  # make its entry in the packet size dictionary
        ##
        labels[key] = label
        ##
        main_inter_arrival_time[key] = []  # create a blank list in this dictionary, as it is the first packet

        last_time[key] = time

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("NOW: COMPUTING AND WRITING FLOW FEATURES INTO CSV...")
header = "Flow ID,Min Packet Length,Max Packet Length,Packet Length Mean,Packet Length Total,Packet Count,Current Packet Length,Flow IAT Min,Flow IAT Max,Flow IAT Mean,Flow Duration,Label"

with open(filename_out, "w") as text_file:
    text_file.write(header)
    text_file.write("\n")
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate features related to packet size
    for key in flow_list:
        packet_list = main_packet_size[key]  # packet_list contains the list of packet sizes for the flow in consideration
        length = len(packet_list)  # number of packets
        avg_pkt_sizes[key] = sum(packet_list) / length  # calculate avg packet size, and store
        min_pkt_size = min(packet_list)
        max_pkt_size = max(packet_list)

        string[key] = key + "," + str(min_pkt_size) + "," + str(max_pkt_size) + "," + str(avg_pkt_sizes[key]) + "," + str(sum(packet_list)) + "," + str(len(packet_list)) + "," + str(packet_list[len(packet_list)-1]) # concatenate features in string format
    # ------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------
    # Now calculate IAT-related features
        inter_arrival_time_list = main_inter_arrival_time[key]  # a list containing IATs for the flow
        length = len(inter_arrival_time_list)
        if length == 0:
            min_IAT = 0
            max_IAT = 0
        else:
            min_IAT = min(inter_arrival_time_list)
            min_IAT_ms = round(1000000000*min_IAT, 9) # convert in nanoseconds
            max_IAT = max(inter_arrival_time_list)
            max_IAT_ms = round(1000000000*max_IAT, 9) # convert in nanoseconds

        if length > 0:
            flow_duration = sum(inter_arrival_time_list)  # flow duration seconds
            flow_duration_ms = round(1000000000*flow_duration, 9) # convert in nanoseconds
            avg_iat = flow_duration / length  # Average IAT
            avg_iat_in_ms = round(1000000000*avg_iat, 9)  # convert in nanoseconds

        if(len(main_packet_size[key]) >= min_number_of_packets):
            string[key] = string[key] + "," + str(min_IAT_ms) + "," + str(max_IAT_ms) + "," + str(avg_iat_in_ms) + "," + str(flow_duration_ms)
            string[key] = string[key] + "," + str(labels[key])
            text_file.write(string[key])
            text_file.write("\n")