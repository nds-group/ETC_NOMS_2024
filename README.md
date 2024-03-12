# Encrypted Traffic Classification at Line Rate in Programmable Switches with Machine Learning

This repository contains the source code for our work on Encrypted Traffic Classification (ETC) in programmable switches with P4 and Machine Learning, appearing in the Proceedings of IEEE/IFIP NOMS 2024, 6–10 May 2024, Seoul, South Korea.

## Overview of the ETC framework
<img src="etc_framework.png" alt="ETC Overview" style="height: 350px; width:500px;"/>  

This work leverages recent advances in data plane programmability to achieve real-time ETC in programmable switches at line rate, with high throughput and low latency. The proposed solution comprises (i) an ETC-aware Random Forest (RF) modelling process where only features based on packet size and packet arrival times are used, and (ii) an encoding of the trained RF model into production-grade P4-programmable switches.

For full details, please consult [our paper](https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/1791/etc_noms24_postprint.pdf?sequence=1&isAllowed=y).

## Organization of the repository  
There are two folders:  
<!-- - _Data_ : information on how to access the data  -->
- _Switch_ : the P4 code for the Tofino switch, the M/A table entries, and the runtime controller code.
- _Offline_ : the jupyter notebooks for training the machine learning models and for offline evaluation, and the scripts for generating the M/A table entries from trained models.

## Use cases
The use cases considered in the paper are: 
- QUIC traffic classification based on the publicly available <a href="https://drive.google.com/drive/folders/1cwHhzvaQbi-ap8yfrj2vHyPmUTQhaYOj">Netflow QUIC dataset</a>. The challenge is classifying traffic into one of 5 classes. 
- Encrypted instant messaging application fingerprinting with 6 classes, based on the <a href="https://ieee-dataport.org/documents/encrypted-mobile-instant-messaging-traffic-dataset">Encrypted Instant Messaging Dataset</a> made available by the NIMS Lab.
- VPN traffic classification, distinguishing 7 classes. It is based on the <a href="https://www.unb.ca/cic/datasets/vpn.html">ISCX-VPN-NonVPN-2016 Dataset</a>.

We provide the python and P4 code for the Encrypted Instant Messaging App classification use case with 6 classes. <br> The same approach for feature/model selection and encoding to P4 applies to all the use cases.

## Citation
If you make use of this code, kindly cite our paper:  
```
@inproceedings{etc-noms-2024,
author = {Akem, Aristide Tanyi-Jong and Fraysse, Guillaume and Fiore, Marco},
title = {Encrypted Traffic Classification at Line Rate in Programmable Switches with Machine Learning},
year = {2024},
booktitle = {Proceedings of NOMS 2024 - IEEE/IFIP Network Operations and Management Symposium},
numpages = {9},
location = {Seoul, South Korea},
series = {NOMS 2024}
}
```

If you need any additional information, send us an email at _aristide.akem_ at _imdea.org_.




