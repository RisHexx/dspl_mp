# -*- coding: utf-8 -*-
"""
Class label mapping for the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
Maps 43 class IDs to human-readable traffic sign names.
"""

SIGN_NAMES = {
    0: "Speed Limit 20 km/h",
    1: "Speed Limit 30 km/h",
    2: "Speed Limit 50 km/h",
    3: "Speed Limit 60 km/h",
    4: "Speed Limit 70 km/h",
    5: "Speed Limit 80 km/h",
    6: "End of Speed Limit 80 km/h",
    7: "Speed Limit 100 km/h",
    8: "Speed Limit 120 km/h",
    9: "No Passing",
    10: "No Passing for Vehicles over 3.5 tons",
    11: "Right-of-Way at Next Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "Vehicles over 3.5 tons Prohibited",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve to the Left",
    20: "Dangerous Curve to the Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows on the Right",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of All Speed and Passing Limits",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout Mandatory",
    41: "End of No Passing",
    42: "End of No Passing by Vehicles over 3.5 tons",
}

NUM_CLASSES = len(SIGN_NAMES)
