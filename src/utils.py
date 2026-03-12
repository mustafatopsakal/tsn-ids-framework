# -*- coding: utf-8 -*-
"""
Utility functions for converting node identifiers and MAC addresses
to their numeric representations in the TSN in-vehicle network topology.
"""


NODE_MAP = {
    "Cam1": 1,
    "Cam2": 2,
    "Cam3": 3,
    "DA_Cam": 4,
    "HU": 5,
    "RSE": 6,
    "Telematics": 7,
    "CU": 8,
    "CD_DVD": 9,
    "Cam4": 10,
    "SW1": 11,
    "SW2": 12,
}


def mac_to_decimal(mac_address):
    """Convert a dash-separated MAC address string to an integer."""
    return int("".join(mac_address.split("-")), 16)


def node_to_decimal(name):
    """Convert a TSN node name to its numeric identifier.

    Raises ValueError if the name is not recognized.
    """
    for key, value in NODE_MAP.items():
        if key in name:
            return value
    raise ValueError(f"Unknown node name: {name}")
