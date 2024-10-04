import pandas as pd
import math
import json

raw_spine_data = pd.read_csv('../data/pni_synapses_v185.csv')
raw_mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')

def nearest_spine_coords(neuron_id, mito_id):
    spine_rows = raw_spine_data.loc[raw_spine_data.post_root_id == neuron_id]
    mito_neuron_df = raw_mito_data[raw_mito_data.cellid == neuron_id]
    mito_row = mito_neuron_df.loc[mito_neuron_df.mito_id == mito_id].iloc[0]

    mito_ctr_x, mito_ctr_y, mito_ctr_z = mito_row.ctr_pos_x_vx * 0.004, mito_row.ctr_pos_y_vx * 0.004, mito_row.ctr_pos_z_vx * 0.04

    spine_data = {}
    for spine_row in spine_rows.itertuples(index=False):
        spine_data[int(spine_row.id)] = (float(spine_row.post_pos_x_vx * 0.004), float(spine_row.post_pos_y_vx * 0.004), float(spine_row.post_pos_z_vx * 0.04))

    distances_to_spines = {} # in voxels
    for spine_id, ctr_coords in spine_data.items():
        #do something with the spine id eventually
        spine_ctr_x, spine_ctr_y, spine_ctr_z = ctr_coords[0], ctr_coords[1], ctr_coords[2]
        spine_distance = math.sqrt(((spine_ctr_x - mito_ctr_x)**2) + ((spine_ctr_y - mito_ctr_y)**2) + ((spine_ctr_z - mito_ctr_z)**2)) # input units are in voxels and output distance is in um
        distances_to_spines[spine_distance] = spine_id

    nearest_spine_distance = min(distances_to_spines.keys())
    nearest_spine_id = distances_to_spines[nearest_spine_distance]
    nearest_spine_ctr_x, nearest_spine_ctr_y, nearest_spine_ctr_z = spine_data[nearest_spine_id]

    return nearest_spine_ctr_x, nearest_spine_ctr_y, nearest_spine_ctr_z

def curves_within_range(neuron_id, mito_id, nearest_spine_ctr_x, nearest_spine_ctr_y, nearest_spine_ctr_z):
    with open(f'../data/jsons/{neuron_id}-{mito_id}-curvature_data.json', 'r') as f:
        raw_curve_data = json.load(f)
        curve_data = [tuple(l) for l in raw_curve_data]

    all_distances_to_curves = {}
    include_curves = {}
    exclude_curves = {}
    radius = 2 # in um
    for curve in curve_data:
        curvature, curve_x, curve_y, curve_z = curve[0], curve[1], curve[2], curve[3]
        curve_distance = math.sqrt(((nearest_spine_ctr_x - curve_x)**2) + ((nearest_spine_ctr_y - curve_y)**2) + ((nearest_spine_ctr_z - curve_z)**2)) # units in um
        if curve_distance < radius:
            include_curves[curve_distance] = curvature
        else:
            exclude_curves[curve_distance] = curvature
        all_distances_to_curves[curve_distance] = curvature
    print(include_curves, exclude_curves)

mito_list = []
def vertex_curvatures_ratio(neuron_id, mito_id):
    with open(f'../data/jsons/{neuron_id}-{mito_id}-curvature_data.json', 'r') as f:
        raw_curve_data = json.load(f)
        curve_data = [tuple(l) for l in raw_curve_data]

    vertices_above_threshold = 0
    total_vertices = 0
    for curve in curve_data:
        curvature = curve[0]
        if curvature < -25:
            vertices_above_threshold += 1
        total_vertices += 1
    ratio = vertices_above_threshold / total_vertices
    
    if (ratio > 0.05 and ratio < 0.1) and neuron_id == 648518346349537516:
        mito_list.append(mito_id)
    
    return ratio, mito_list