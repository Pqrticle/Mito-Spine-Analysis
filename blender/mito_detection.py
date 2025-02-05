import pandas as pd
import numpy as np
import json
import trimesh
from scipy.spatial import KDTree

def detect_nearby_mito(neuron_id, mito_meshes):
    filtered_synapse_data = pd.read_csv('../data/filtered_synapse_data.csv')
    matching_rows = filtered_synapse_data[filtered_synapse_data['post_root_id'] == neuron_id]
    matched_base_coords = np.array(matching_rows['base_coords'].apply(json.loads).tolist(), dtype=np.float64)

    # Build a KDTree for all mitochondrial mesh vertices
    mito_vertices = []
    vertex_to_mesh_map = []
    for mito_id, mito_mesh in mito_meshes.items():
        mito_mesh.vertices /= 1000
        mito_vertices.append(mito_mesh.vertices)
        vertex_to_mesh_map.extend([mito_id] * len(mito_mesh.vertices))
    
    # Combine all vertices into a single array for KDTree
    all_vertices = np.vstack(mito_vertices)
    vertex_tree = KDTree(all_vertices)

    # Process each marked point
    for base_coord in matched_base_coords:
        # Find indices of vertices within the threshold
        nearby_vertex_indices = vertex_tree.query_ball_point(base_coord, r=1)
        nearby_mito_ids = list(set(vertex_to_mesh_map[idx] for idx in nearby_vertex_indices))
        
        if not nearby_mito_ids:
            nearby_mito_ids = 'N/A'
            intersected_volumes = 'N/A'
            total_volume = 'N/A'      
        else:
            intersected_volumes, total_volume = find_intersected_volumes(nearby_mito_ids, mito_meshes, base_coord)

        base_coord, nearby_mito_ids, intersected_volumes, total_volume = json.dumps(base_coord.tolist()), json.dumps(nearby_mito_ids), json.dumps(intersected_volumes), json.dumps(total_volume)
        synapse_index = filtered_synapse_data[filtered_synapse_data['base_coords'] == base_coord].index
        filtered_synapse_data.loc[synapse_index, 'mito_ids'] = nearby_mito_ids
        filtered_synapse_data.loc[synapse_index, 'intersected_volumes'] = intersected_volumes
        filtered_synapse_data.loc[synapse_index, 'total_intersected_volume'] = total_volume

    filtered_synapse_data.to_csv('../data/filtered_synapse_data.csv', index=False)
    matching_rows = filtered_synapse_data[filtered_synapse_data['post_root_id'] == neuron_id]
    print(matching_rows)

def find_intersected_volumes(nearby_mito_ids, mito_meshes, base_coord):
    sphere = trimesh.creation.icosphere(radius=1)
    sphere.apply_translation(base_coord)
    total_volume = 0
    intersected_volumes = []
    for nearby_id in nearby_mito_ids:
        nearby_mesh = mito_meshes[nearby_id]
        if nearby_mesh.is_volume:
            try:
                intersection = sphere.intersection(nearby_mesh)
                if isinstance(intersection, trimesh.Trimesh) and not intersection.is_empty:
                    total_volume += intersection.volume
                    intersected_volumes.append(intersection.volume)
            except Exception as e:
                intersected_volumes.append('N/A')
                print(f"Intersection failed for Base Coordinate {base_coord} and Mito ID {nearby_id}: {e}")
    return intersected_volumes, total_volume