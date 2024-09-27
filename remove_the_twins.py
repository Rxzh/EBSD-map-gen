import pymicro
from pymicro.crystal.microstructure import Orientation

from tqdm import tqdm
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#from datasets import load_dataset, load

from pymicro.crystal.ebsd import OimScan
from pymicro.crystal.lattice import Lattice, CrystallinePhase, Symmetry
from pymicro.crystal.microstructure import Microstructure
from pymicro.crystal.quaternion import Quaternion

def find(parent, i):
    if parent[i] == i:
        return i
    else:
        # Path compression heuristic
        parent[i] = find(parent, parent[i])
        return parent[i]

def union(parent, size, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    
    if xroot != yroot:
        # Always attach the smaller tree to the larger tree based on size
        if size[xroot] < size[yroot]:
            parent[xroot] = yroot
            size[yroot] += size[xroot]
        else:
            parent[yroot] = xroot
            size[xroot] += size[yroot]

def merge_twins(twins_pairs, grains_ids):
    parent = {}
    size = {}

    # Extract unique grain IDs from the 2D grains_ids map
    unique_grains = np.unique(grains_ids)

    # Initialize the parent and size for each unique grain ID
    for grain_id in unique_grains:
        parent[grain_id] = grain_id
        size[grain_id] = np.sum(grains_ids == grain_id)

    # Process each pair and union the sets
    for grain_id, neighbor_id in twins_pairs:
        union(parent, size, grain_id, neighbor_id)

    # Create a new grains_ids map by updating each pixel to its representative
    new_grains_ids_map = np.copy(grains_ids)
    for i in range(new_grains_ids_map.shape[0]):
        for j in range(new_grains_ids_map.shape[1]):
            new_grains_ids_map[i, j] = find(parent, new_grains_ids_map[i, j])

    return new_grains_ids_map


def is_sigma_3_twin(orientation1, orientation2, crystal_structure=Symmetry.cubic):
    """Check if two orientations form a Σ3 twin relationship (60° rotation around <111> axis).

    :param orientation1: an instance of Orientation class
    :param orientation2: an instance of Orientation class
    :returns bool: True if the orientations form a Σ3 twin, False otherwise.
    """
    sigma_3_angle = np.pi / 3  # 60 degrees in radians
    sigma_3_axis = np.array([1, 1, 1]) / np.sqrt(3)  # <111> axis

    symmetries = crystal_structure.symmetry_operators()
    (gA, gB) = (orientation1.orientation_matrix(), orientation2.orientation_matrix())

    for (g1, g2) in [(gA, gB), (gB, gA)]:
        for j in range(symmetries.shape[0]):
            sym_j = symmetries[j]
            oj = np.dot(sym_j, g1)  # the crystal symmetry operator is left applied
            for i in range(symmetries.shape[0]):
                sym_i = symmetries[i]
                oi = np.dot(sym_i, g2)
                delta = np.dot(oi, oj.T)
                mis_angle = Orientation.misorientation_angle_from_delta(delta)
                mis_axis = Orientation.misorientation_axis_from_delta(delta)
                if np.isclose(mis_angle, sigma_3_angle, atol=np.deg2rad(15)) and np.allclose(mis_axis, sigma_3_axis, atol=0.02):
                    return True
    return False

def are_twins(m, grain1_id, grain2_id):

    # Retrieve orientation matrices for both grains
    grain1 = m.get_grain(grain1_id)
    grain2 = m.get_grain(grain2_id)
    
    orientation1 = Orientation(m.get_grain(grain1_id).orientation_matrix())
    orientation2 = Orientation(m.get_grain(grain2_id).orientation_matrix())
    
    return is_sigma_3_twin(orientation1, orientation2)


if not os.path.exists('data/grains_no_twins_ids_maps'):
    os.mkdir('data/grains_no_twins_ids_maps')

for fname in tqdm(os.listdir('data/quat_ebsd_maps/')):
    if os.path.exists(os.path.join('data/grains_no_twins_ids_maps', fname)):
        continue

    # load the segmentation
    grain_ids = np.squeeze(np.load(os.path.join('data/grains_ids_maps',fname)))

    # directly import the EBSD data, here we chose not to take the confidence index into account
    m = Microstructure.from_ebsd(os.path.join('data/raw_ebsd_scans',fname.replace('.npy','.ctf')),  grain_ids=grain_ids)
    m.dilate_grains(dilation_steps=3, new_map_name='grain_map', update_microstructure_properties=True)
    grain_ids_dilated = np.squeeze(m.get_grain_map())
    m.quats = np.load(os.path.join('data/quat_ebsd_maps', fname))

    print('Searching for twins...')
    twins_pairs=[]
    for grain in tqdm(m.grains):
        grain_id=grain['idnumber']
        if not(grain_id==0):
            for neighbor_id in m.find_neighbors(grain_id):
                if not(neighbor_id==0):
                    if are_twins(m, grain_id, neighbor_id):
                        #print('grain: {} and neighbor: {} are twins'.format(grain_id, neighbor_id))
                        twins_pairs.append([grain_id, neighbor_id])

    new_grain_ids = merge_twins(twins_pairs,grain_ids_dilated)

    np.save(os.path.join('data/grains_no_twins_ids_maps', fname), new_grain_ids)

