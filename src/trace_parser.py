# -*- coding: utf-8 -*-
"""
Parse raw OMNeT++ TSN simulation trace files (.txt) and convert them
into labeled CSV files suitable for the ML pipeline.

Trace files are tab-separated and contain lines like:
    event_no  timestamp  source --> dest  Stream X  ...  mac_info  packet_info

The parser filters lines containing "Stream", extracts relevant fields,
and assigns labels based on manipulated stream IDs encoded in the filename.
"""

import os
import re


def parse_manipulated_streams(filename):
    """Extract manipulated stream IDs from an attack scenario filename.

    Expected naming: as1_streams(21)_periods(9.5)_45sec.txt
    Returns a set of ints, e.g. {21}.
    """
    match = re.search(r"streams\((.*?)\)", filename)
    if match:
        return set(map(int, re.findall(r"\d+", match.group(1))))
    return set()


def parse_trace_file(file_path, manipulated_streams):
    """Parse a single OMNeT++ trace file and return structured data.

    Parameters
    ----------
    file_path : str
        Path to the tab-separated trace file.
    manipulated_streams : set of int
        Stream IDs that are considered malicious. Pass an empty set for
        normal (non-attack) scenarios.

    Returns
    -------
    data : list of list
        Each row: [timestamp, stream_no, source, destination,
                   source_mac, dest_mac, packet_size]
    labels : list of int
        0 = benign, 1 = malicious
    count_zeros : int
    count_ones : int
    """
    data = []
    labels = []
    count_zeros = 0
    count_ones = 0

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 7 or "Stream" not in parts[3]:
                continue

            timestamp = parts[1]
            stream_no = int(parts[3].split()[1])
            source_dest = parts[2].split(" --> ")
            source = source_dest[0]
            destination = source_dest[1]
            source_mac = parts[5].split()[1]
            dest_mac = parts[5].split()[3]
            packet_size = parts[6].split("::")[1].split(":")[1].split(" ")[0]

            label = 1 if stream_no in manipulated_streams else 0

            data.append([timestamp, stream_no, source, destination,
                         source_mac, dest_mac, packet_size])
            labels.append(label)
            if label == 1:
                count_ones += 1
            else:
                count_zeros += 1

    return data, labels, count_zeros, count_ones


def write_csv(data, labels, output_path):
    """Write parsed data and labels to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w+") as f:
        for i in range(len(data)):
            f.write(",".join(map(str, data[i] + [labels[i]])) + "\n")


def process_trace_directory(trace_dir, output_dir):
    """Process all trace files in a directory and write CSVs.

    Parameters
    ----------
    trace_dir : str
        Root directory containing normal/ and attack/ subdirectories
        with .txt trace files.
    output_dir : str
        Root directory where CSV files will be written into
        normal/ and attack/ subdirectories.
    """
    normal_dir = os.path.join(trace_dir, "normal")
    attack_dir = os.path.join(trace_dir, "attack")

    if os.path.isdir(normal_dir):
        for fname in os.listdir(normal_dir):
            if not fname.endswith(".txt"):
                continue
            print(f"Parsing normal trace: {fname}")
            fpath = os.path.join(normal_dir, fname)
            data, labels, zeros, ones = parse_trace_file(fpath, set())
            csv_name = os.path.splitext(fname)[0] + ".csv"
            write_csv(data, labels, os.path.join(output_dir, "normal", csv_name))
            print(f"  Zeros: {zeros}  Ones: {ones}  Total: {zeros + ones}")

    if os.path.isdir(attack_dir):
        for fname in sorted(os.listdir(attack_dir)):
            if not fname.endswith(".txt"):
                continue
            streams = parse_manipulated_streams(fname)
            print(f"Parsing attack trace: {fname}  (manipulated streams: {streams})")
            fpath = os.path.join(attack_dir, fname)
            data, labels, zeros, ones = parse_trace_file(fpath, streams)
            csv_name = os.path.splitext(fname)[0] + ".csv"
            write_csv(data, labels, os.path.join(output_dir, "attack", csv_name))
            print(f"  Zeros: {zeros}  Ones: {ones}  Total: {zeros + ones}")
