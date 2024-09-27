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
from pymicro.crystal.microstructure import Microstructure, Orientation
from pymicro.crystal.quaternion import Quaternion




def main(fname):

        MEAN_ORIENTATIONS = dict() #keys are IDs, values are quaternions (np.array)

        if os.path.exists(os.path.join('data/mean_quats_maps_no_twins/',fname,)):
            print('mean after twins removal exists for: ', fname)
            return


        if not(os.path.exists(os.path.join('data/grains_no_twins_ids_maps/',fname))):
            print('grain ids maps after twins removal not found for: ', fname)
            return


        # load the segmentation
        grain_ids = np.squeeze(np.load(os.path.join('data/grains_ids_maps',fname)))
        grain_ids_no_twins = np.squeeze(np.load(os.path.join('data/grains_no_twins_ids_maps/',fname)))

        # directly import the EBSD data, here we chose not to take the confidence index into account
        m = Microstructure.from_ebsd(os.path.join('data/raw_ebsd_scans',
                                                  fname.replace('.npy','.ctf')),  
                                     grain_ids=grain_ids)

        m.dilate_grains(dilation_steps=3, new_map_name='grain_map', update_microstructure_properties=True)
        grain_ids_dilated = np.squeeze(m.get_grain_map())

        m.quats = np.load(os.path.join('data/quat_ebsd_maps', fname))

        rods = Orientation.eu2ro(m.get_field('euler').reshape((m.get_grain_map().shape[0]*m.get_grain_map().shape[1], 3)))
        rods = rods.reshape((m.get_grain_map().shape[0],m.get_grain_map().shape[1], 3))

        m.set_orientation_map(rods)
        print('orientation_map is none? : ', rods is None)

        sym = m.get_phase(1).get_symmetry()
        for gid in np.unique(grain_ids_no_twins):
            
            rods_gid = m.fz_grain_orientation_data(gid, plot=False, move_to_fz=True)
            mean_quat = Orientation.compute_mean_orientation(rods_gid, symmetry=sym).quat.quat
            MEAN_ORIENTATIONS[gid] = mean_quat

        # Initialize the new map with the desired shape (n, m, 4)
        new_map = np.zeros((grain_ids_no_twins.shape[0], grain_ids_no_twins.shape[1], 4))

        # Loop over the grain_ids_no_twins array and populate the new map
        for i in range(grain_ids_no_twins.shape[0]):
            for j in range(grain_ids_no_twins.shape[1]):
                grain_id = grain_ids_no_twins[i, j]
                new_map[i, j] = MEAN_ORIENTATIONS[grain_id]

        np.save(os.path.join('data/mean_quats_maps_no_twins/',fname), new_map)
        
if __name__ == '__main__':
    main(sys.argv[1])


