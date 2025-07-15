import pandas as pd
import numpy as np
import os
import json
import trimesh
import dendrite_identifier
from scipy.spatial import KDTree

mesh_error_count = 0
synapse_count = 0

def generate_mito_meshes(neuron_id, mito_mm, mito_mesh_dir):
    raw_mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
    
    # Filter for mito within dendrites of the neuron
    all_mito_df = raw_mito_data[raw_mito_data.cellid == neuron_id]
    dendritic_mito_df = all_mito_df[all_mito_df.compartment.isin(['Basal', 'Apical', 'Unknown dendritic'])]
    mito_ids = dendritic_mito_df.mito_id.tolist()
  
    # Determine if mito are already cached
    mito_meshes = {}
    for mito_id in mito_ids:
        mito_id_path = f'{mito_mesh_dir}/{mito_id}.h5'
        if not os.path.exists(mito_id_path):
            mito_mesh = mito_mm.mesh(seg_id=mito_id, remove_duplicate_vertices=True)
            print(f"Loaded mitochondria mesh: {mito_id}")
        else:
            mito_mesh = mito_mm.mesh(filename=mito_id_path)
        mito_meshes[mito_id] = mito_mesh
    
    return mito_meshes

def find_intersected_volumes(nearby_mito_ids, mito_meshes, base_coord, radius):
    global mesh_error_count, synapse_count
    sphere = trimesh.creation.icosphere(radius)
    sphere.apply_translation(base_coord)
    total_intersected_volume = 0
    intersected_volumes = []
    for nearby_id in nearby_mito_ids:
        nearby_mesh = mito_meshes[nearby_id]
        if nearby_mesh.is_volume:
            intersection = sphere.intersection(nearby_mesh)
            if isinstance(intersection, trimesh.Trimesh) and not intersection.is_empty:
                total_intersected_volume += intersection.volume
                intersected_volumes.append(intersection.volume)
            else:
                intersected_volumes.append("MeshError")
        else:
            intersected_volumes.append("MeshError")
    if "MeshError" in intersected_volumes:
        total_intersected_volume = "NaN"
        mesh_error_count += 1
    synapse_count += 1

    return intersected_volumes, total_intersected_volume

mito_volume_data = {}
def detect_nearby_mito(neuron_id, neuron_mesh, mito_meshes, osi, dsi, radius):
    synapse_table = pd.read_csv('../data/synapse_table.csv')
    synapse_df = synapse_table[synapse_table['post_root_id'] == neuron_id]
    spine_bases = synapse_df['base_coords'].apply(json.loads).tolist()

    # Build a KDTree for all mitochondrial mesh vertices
    combined_mito_volume = 0
    mito_vertices = []
    vertex_to_mesh_map = []
    for mito_id, mito_mesh in mito_meshes.items():
        mito_mesh.vertices /= 1000
        mito_vertices.append(mito_mesh.vertices)
        vertex_to_mesh_map.extend([mito_id] * len(mito_mesh.vertices))
        combined_mito_volume += mito_mesh.volume
    
    all_vertices = np.vstack(mito_vertices)
    vertex_tree = KDTree(all_vertices)

    # Process each marked point
    spine_base_volumes = []
    for base_coord in spine_bases:
        nearby_vertex_indices = vertex_tree.query_ball_point(base_coord, radius)
        nearby_mito_ids = list(set(vertex_to_mesh_map[idx] for idx in nearby_vertex_indices))
        
        if nearby_mito_ids: 
            intersected_volumes, total_intersected_volume = find_intersected_volumes(nearby_mito_ids, mito_meshes, base_coord, radius)
            spine_base_volumes.append(total_intersected_volume)
            nearby_mito_ids = json.dumps(nearby_mito_ids)
            intersected_volumes = json.dumps(intersected_volumes)
        else:
            intersected_volumes = [0]
            total_intersected_volume = 0
            spine_base_volumes.append(total_intersected_volume)
            nearby_mito_ids = ["NoMito"]
    
        synapse_index = synapse_df[synapse_df['base_coords'].apply(json.loads).apply(lambda x: x == base_coord)].index[0]
        synapse_table.loc[synapse_index, 'mito_ids'] = nearby_mito_ids
        synapse_table.loc[synapse_index, 'intersected_volumes'] = intersected_volumes
        synapse_table.loc[synapse_index, 'total_intersected_volume'] = [total_intersected_volume]
        synapse_table.loc[synapse_index, 'osi_value'] = [osi]
        synapse_table.loc[synapse_index, 'dsi_value'] = [dsi]

    print('Computed all intraneuronal mitochondria volume data')
    dendritic_volume = dendrite_identifier.calculate_dendritic_volume(neuron_id, neuron_mesh)
    mito_volume_data[neuron_id] = [osi, dsi, dendritic_volume, combined_mito_volume, spine_base_volumes]
    with open(f'../data/mito_volume_data_r={radius}.json', 'w') as jsonfile:
        json.dump(mito_volume_data, jsonfile, indent=4)
    
    synapse_table.to_csv(f'../data/synapse_table.csv', index=False)

    with open('../data/mesh_error_count.txt', 'w') as f:
        f.write(f"Synapses with Mitochondria Mesh Errors: {mesh_error_count}\n")
        f.write(f"Total Synapse Count: {synapse_count}\n")
        f.write(f"Rate of Error: {mesh_error_count * 100 / synapse_count}%")