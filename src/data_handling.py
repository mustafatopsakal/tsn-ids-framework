# -*- coding: utf-8 -*-
"""
Data loading, feature extraction, and output writing for the
CSV-based ML pipeline.

Reads labeled CSV files (7 features + label), applies feature extraction
(inter-arrival time + stream history), and produces ML-ready datasets
(11 features + label).
"""

from utils import mac_to_decimal, node_to_decimal

FE_START_INDEX = 3  # sliding window depth for feature extraction


def load_csv(file_path):
    """Load a labeled CSV file and return structured data.

    Each CSV line: timestamp,stream_id,source,destination,
                   source_mac,dest_mac,packet_length,label

    Returns
    -------
    data : list of list
        Each row: [timestamp, stream_no, source, destination,
                   source_mac, dest_mac, packet_length]
    labels : list of int
    count_zeros : int
    count_ones : int
    """
    data = []
    labels = []
    count_zeros = 0
    count_ones = 0

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            timestamp = float(parts[0])
            stream_no = int(parts[1])
            source = node_to_decimal(parts[2])
            destination = node_to_decimal(parts[3])
            source_mac = mac_to_decimal(parts[4])
            dest_mac = mac_to_decimal(parts[5])
            packet_length = int(parts[6])
            label = int(parts[7])

            data.append([timestamp, stream_no, source, destination,
                         source_mac, dest_mac, packet_length])
            labels.append(label)
            if label == 1:
                count_ones += 1
            else:
                count_zeros += 1

    return data, labels, count_zeros, count_ones


def feature_extraction(data):
    """Compute temporal features from raw data rows.

    For each row (starting from FE_START_INDEX), computes:
    - inter-arrival time (difference from previous timestamp)
    - previous 3 stream IDs (sliding window)

    Returns a list of 11-element feature vectors.
    """
    fe_data = []

    for i in range(FE_START_INDEX, len(data)):
        timestamp = data[i][0]
        last_remote_timestamp = float(timestamp) - float(data[i - 1][0])
        stream_id = data[i][1]
        prev_stream_id = data[i - 1][1]
        prev_prev_stream_id = data[i - 2][1]
        prev_prev_prev_stream_id = data[i - 3][1]
        source_id = data[i][2]
        destination_id = data[i][3]
        packet_source_mac = data[i][4]
        packet_dest_mac = data[i][5]
        packet_length = data[i][6]

        fe_data.append([
            timestamp, last_remote_timestamp,
            stream_id, prev_stream_id, prev_prev_stream_id,
            prev_prev_prev_stream_id,
            source_id, destination_id,
            packet_source_mac, packet_dest_mac, packet_length,
        ])

    return fe_data


def write_output(data, labels, output_file):
    """Write feature data and labels to a CSV file."""
    with open(output_file, "w+") as f:
        for i in range(len(data)):
            f.write(",".join(map(str, data[i] + [labels[i]])) + "\n")
