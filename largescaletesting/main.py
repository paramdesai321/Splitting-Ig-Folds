#import dist_from_cm
#import angle_of_strand_from_plane
#import angle_between_strands
#import angle_between_strand_projections
#import length_of_strands
import hydrophobicity
from feature_vector import super_dict
import sys
PIN = sys.argv[1]
# Chronological module order
FEATURE_ORDER = [
    "angle_between_strands",
    "angle_between_strand_projections",
    "angle_of_strand_from_plane",
    "dist_from_cm",
    "length_of_strands",
    "hydrophobicity"
]
# Detailed feature groups for sorting (based on your names)
FEATURE_GROUPS = {
    "angle_between_strands": [
        "Angle between B and C",
        "Angle between B and E",
        "Angle between B and F",
        "Angle between C and E",
        "Angle between C and F",
        "Angle between E and F",
    ],
    "angle_between_strand_projections": [
        "Angle between the projections of B and C on the z=0 plane",
        "Angle between the projections of B and F on the z=0 plane",
        "Angle between the projections of C and E on the z=0 plane",
        "Angle between the projections of E and F on the z=0 plane",
    ],
    "angle_of_strand_from_plane": [
        "Angle of B strand vector to x axis",
        "Angle of B strand vector to y axis",
        "Angle of B strand vector to z axis",
        "Angle of C strand vector to x axis",
        "Angle of C strand vector to y axis",
        "Angle of C strand vector to z axis",
        "Angle of E strand vector to x axis",
        "Angle of E strand vector to y axis",
        "Angle of E strand vector to z axis",
        "Angle of F strand vector to x axis",
        "Angle of F strand vector to y axis",
        "Angle of F strand vector to z axis",
    ],
    "dist_from_cm": [
        "Distance of Center of Mass of B from plane",
        "Distance of Center of Mass of C from plane",
        "Distance of Center of Mass of E from plane",
        "Distance of Center of Mass of F from plane",
    ],
    "length_of_strands": [
        "Length of B Strand Vector",
        "Length of C Strand Vector",
        "Length of E Strand Vector",
        "Length of F Strand Vector",
    ],
    "hydrophobicity": [
        "NonPolar",
        "Ala",
        "Pro",
        "Gly",
        "Aromatic",
        "Polar",
        "Negative",
        "Positive",
        "Cys",
    ],
}

def sort_features_by_module_order(features: dict) -> dict:
    ordered = {}
    for module in FEATURE_ORDER:
        for name in FEATURE_GROUPS[module]:
            if name in features:
                ordered[name] = features[name]
    # add any leftover features (e.g., new ones) at the end
    for name in features:
        if name not in ordered:
            ordered[name] = features[name]
    return ordered
ordered_features = sort_features_by_module_order(super_dict[PIN])
#print(ordered_features)
super_dict[PIN] = ordered_features
to_remove = {"NonPolar", "Ala", "Pro", "Gly", "Aromatic", "Polar", "Negative", "Positive", "Cys"}

for pin, feats in super_dict.items():
    super_dict[pin] = {k: v for k, v in feats.items() if k not in to_remove}

#print(super_dict[PIN].keys())
#print("final super_dict:", super_dict)
#del super_dict['1A4K_L_3_to_107']['']
#print(super_dict[PIN])
print(len(super_dict[PIN].items()) )


