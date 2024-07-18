# ENCRYPTED TRAFFIC CLASSIFICATION

This repo contains the scripts used for the data preparation and data engineering used in the paper [Encrypted Traffic Classification at Line Rate in Programmable Switches with Machine Learning](https://dspace.networks.imdea.org/handle/20.500.12761/1791) by Aristide Tanyi-Jong Akem, Guillaume Fraysse, Marco Fiore presented at IEEE/IFIP Network Operations and Management Symposium (NOMS) 2024.

## Datasets
5 different datasets containing Encrypted data were considered. Some are public some require a subscription to access them. The first three were kept in the paper:
* The [ISCXVPN2016 Dataset](http://dx.doi.org/10.5220/0005740704070414) dataset (from Draper-Gil, G.; Lashkari, A.; Mamun, M. and A. Ghorbani, A. (2016). Characterization of Encrypted and VPN Traffic using Time-related Features. In Proceedings of the 2nd International Conference on Information Systems Security and Privacy - ICISSP; ISBN 978-989-758-167-0; ISSN 2184-4356, SciTePress, pages 407-414. DOI: 10.5220/0005740704070414) is a popular labeled dataset made available by the Canadian Institute of Cybersecurity (CIC) from University of New Brunswick (UNB). It comprises about 28GB of traffic data captured using tcpdump and Wireshark. A subset of this dataset is made of VPN data, generated using an external VPN service.For this work the dataset was processed
from the raw PCAP files using the pipeline described in Section 2 to keep only the VPN subset and aggregate the packets in flows. This results in 4960 flows. Classes This dataset includes 7 classes of encrypted traffic: Browsing, Email,
Chat, Streaming, File Transfer, VoIP, and P2P. Visualization Figure 6a shows the distribution of the samples
* NOMS2023 Encrypted Mobile Instant Messaging Traffic Dataset. The **NOMS2023 Encrypted Mobile Instant Messaging Traffic Dataset** (by Zolboo Erdenebaatar, Riyad Alshammari, Nur Zincir-Heywood, Marwa Elsayed, Biswajit Nandy, Nabil Seddigh, January 23, 2023, "Encrypted Mobile Instant Messaging Traffic Dataset", IEEE Dataport) can be downloaded at [https://dx.doi.org/10.21227/aer2-kq52](https://dx.doi.org/10.21227/aer2-kq52). It is divided in 7 files in the zip format. Six of these files contains data from traffic to commonly used Instant Messaging applications (Discord, Facebook Messenger, Signal, Microsoft Teams, Telegram and WhatsApp). The last file (non_ima_encrypted_traffic.zip) contains encrypted traffic that is not from any of this classes and is not traffic from Instant Messaging applications. It contains four classes, the first three are other types of usage: Gmail, WebBrowsing, YouTube. The last class Background contains all background
traffic, i.e. traffic recorded during the same period but that is not for the classes identified by the other applications. For this work we considered only the data from the 6 Instant Messaging application and considered the classification
of traffic in these six classes. The subset of the dataset that is then considered contains 6 different classes: Discord, Facebook Messenger, Signal, Microsoft Teams, Telegram and WhatsApp.
* The Netflow QUIC dataset from [V. Tong, H. A. Tran, S. Souihi and A. Mellouk, "A Novel QUIC Traffic Classifier Based on Convolutional Neural Networks," 2018 IEEE Global Communications Conference (GLOBECOM), Abu Dhabi, United Arab Emirates, 2018, pp. 1-6, doi: 10.1109/GLOCOM.2018.8647128.](https://ieeexplore.ieee.org/abstract/document/8647128) is a labeled dataset of QUIC traffic to Google services. This dataset is significantly larger than the others with 365000 flows and a total of 136 millions packets. This dataset contains traffic classified in 5 different classes from Google services: CHAT, VoIP, FileTransfer, Video streaming YouTube, Google Play Music.
* [UC Davis](https://doi.org/10.48550/arXiv.1812.09761) : The UCDavis QUIC Dataset is a labeled dataset that can be downloaded
at [https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE](https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE) (file pretraining.zip). Traffic on different services offered by Google was captured by University of California, Davis (UC Davis) team. The data was collected using AutoIt4 and Selenium WebDriver5 scripts on different systems running various versions of Windows and Ubuntu Linux. Only the QUIC traffic was kept. This dataset contains 5 classes which are 5 different Google Services: Google Drive, YouTube, Google Doc, Google Search and Google Music.
* The CSTNET TLS1.3 dataset (by [Lin, X., Xiong, G., Gou, G., Li, Z., Shi, J. and Yu, J., 2022, April. Et-bert: A contextualized datagram representation with pre-training transformers for encrypted traffic classification. In Proceedings of the ACM Web Conference 2022 (pp. 633-642)](https://dl.acm.org/doi/abs/10.1145/3485447.3512217) is a labeled dataset of encrypted traffic to a large number (120) of services. This high number of classes is an order of magnitude bigger than the other 4 datasets. It is probably more realistic from a network operator perspective whose customers generate traffic not only to a handful of services but to any service on the internet. This dataset contains data from 120 classes, each of which is labeled by the domain name of an application (e.g. google.com, elsevier.com, ..).

## Data preparation
Most datasets are in raw PCAP format. We have performed two steps:
* convert PCAP to CSV
* compute the flows for each packet. A flow is a 5-tuple (IP src, port src, ip dst, port dst, protocol). Each packet with the same value for the tuple get associated with a unique flow-id. This step add a new column in the CSV file with this flow id.

### PCAP to csv
After using a Python script absed on scappy we moved to tshark for performance on the larger datasets.
We used the tshark command, cf. the script **data_preparation/pcap2csv.sh**. Once you have downloaded a dataset you can run the script on each of the PCAP files and redirect the output to a CSV file:

```bash
bash data_preparation/pcap2csv.sh datafile.pcap > datafile.csv
```

### Add the flow id column

To add the flow Id information to the dataset, we have developped the Python script **data_preparation/pkts2flows.py**. 
To use it, simply change the placeholder values *inputdir* and *outputdir* in the script.
* *inputdir* must point to the directory where the csv files of the dataset are stored.
* *outputdir* must point to the folder where you want the new files to be written.

```bash
python data_preparation/pkts2flows.py 
```
