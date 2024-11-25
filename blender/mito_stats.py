import pandas as pd
import math
import json

#raw_spine_data = pd.read_csv('../data/pni_synapses_v185.csv')

def nearest_spine_coords(neuron_id, mito_id):
    #spine_rows = raw_spine_data.loc[raw_spine_data.post_root_id == neuron_id]
    raw_mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
    mito_neuron_df = raw_mito_data[raw_mito_data.cellid == neuron_id]
    mito_row = mito_neuron_df.loc[mito_neuron_df.mito_id == mito_id].iloc[0]
    mito_ctr_x, mito_ctr_y, mito_ctr_z = mito_row.ctr_pos_x_vx * 0.004, mito_row.ctr_pos_y_vx * 0.004, mito_row.ctr_pos_z_vx * 0.04

    with open(f'../data/jsons/{neuron_id}-spine-base-coords', 'r') as file:
        spine_base_coords = json.load(file)

    '''
    spine_data = {}
    for spine_row in spine_rows.itertuples(index=False):
        #spine_data[int(spine_row.id)] = (float(spine_row.post_pos_x_vx * 0.004), float(spine_row.post_pos_y_vx * 0.004), float(spine_row.post_pos_z_vx * 0.04))

    distances_to_spines = {} # in voxels
    for spine_id, ctr_coords in spine_data.items():
        #do something with the spine id eventually
        spine_ctr_x, spine_ctr_y, spine_ctr_z = ctr_coords[0], ctr_coords[1], ctr_coords[2]
        spine_distance = math.sqrt(((spine_ctr_x - mito_ctr_x)**2) + ((spine_ctr_y - mito_ctr_y)**2) + ((spine_ctr_z - mito_ctr_z)**2)) # input units are in voxels and output distance is in um
        distances_to_spines[spine_distance] = spine_id #dont technically need spine id

    nearest_spine_distance = min(distances_to_spines.keys())
    nearest_spine_id = distances_to_spines[nearest_spine_distance]
    nearest_spine_ctr_x, nearest_spine_ctr_y, nearest_spine_ctr_z = spine_data[nearest_spine_id]'''

    distances_to_spines = {} # in voxels
    for base_coords in spine_base_coords:
        spine_base_x, spine_base_y, spine_base_z = base_coords[0], base_coords[1], base_coords[2]
        spine_distance = math.sqrt(((spine_base_x - mito_ctr_x)**2) + ((spine_base_y - mito_ctr_y)**2) + ((spine_base_z - mito_ctr_z)**2)) # input units are in voxels and output distance is in um
        distances_to_spines[spine_distance] = spine_base_x, spine_base_y, spine_base_z

    nearest_spine_distance = min(distances_to_spines.keys())
    nearest_spine_ctr_x, nearest_spine_ctr_y, nearest_spine_ctr_z = distances_to_spines[nearest_spine_distance]
    
    return nearest_spine_ctr_x, nearest_spine_ctr_y, nearest_spine_ctr_z

def curves_within_sphere(curve_data, nearest_spine_ctr_x, nearest_spine_ctr_y, nearest_spine_ctr_z):
    all_distances_to_curves = {}
    include_curves = {}
    exclude_curves = {}
    radius = 1 # in um
    for curve in curve_data:
        curvature, curve_x, curve_y, curve_z = curve[0], curve[1], curve[2], curve[3]
        curve_distance = math.sqrt(((nearest_spine_ctr_x - curve_x)**2) + ((nearest_spine_ctr_y - curve_y)**2) + ((nearest_spine_ctr_z - curve_z)**2)) # units in um
        if curve_distance < radius:
            include_curves[curve_distance] = curvature
        else:
            exclude_curves[curve_distance] = curvature
        all_distances_to_curves[curve_distance] = curvature
    #print(include_curves, exclude_curves)