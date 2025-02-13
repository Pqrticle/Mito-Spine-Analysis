import pandas as pd
import numpy as np
import math
import json
from meshparty import trimesh_io, mesh_filters, trimesh_vtk
from caveclient import CAVEclient
import Guillotine
import skeletor as sk
import activity_clustering

if __name__ == '__main__':
    # Load activity levels and data
    low_activity_levels, high_activity_levels = activity_clustering.activity_levels()
    all_activity_levels = {**low_activity_levels, **high_activity_levels}
    raw_spine_data = pd.read_csv('../data/pni_synapses_v185.csv')
    raw_mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')

    unique_cellids = raw_mito_data['cellid'].unique()
    matches = []
    filtered_psds = {}
    for cellid in unique_cellids:
        snapped_points = []
        microns_points = []
        
        # Load spine base coordinates for the current cell
        with open(f'../data/spine_base_coords/{cellid}-SBC.json', 'r') as file:
            spine_base_coords = json.load(file)
        spine_bases = np.array(spine_base_coords)
        
        matching_rows = raw_spine_data[raw_spine_data['post_root_id'] == cellid]
        psd_coords = matching_rows[['post_pos_x_vx', 'post_pos_y_vx', 'post_pos_z_vx']].values.astype(np.float64)
        psd_coords *= np.array([0.004, 0.004, 0.04])

        # Filter rows with activity levels
        mapped_activity_levels = matching_rows['pre_root_id'].map(all_activity_levels)
        valid_psd_indices = mapped_activity_levels.notnull() # replace with isnull() / notnull()
        psd_coords = psd_coords[valid_psd_indices]
        mapped_activity_levels = mapped_activity_levels[valid_psd_indices].values
        ids = matching_rows[valid_psd_indices]['id'].values
        pre_root_ids = matching_rows[valid_psd_indices]['pre_root_id'].values

        if len(psd_coords) > 0:
            filtered_psds[str(cellid)] = []

        # Match points using a greedy algorithm
        while len(psd_coords) > 0 and len(spine_bases) > 0:
            distances = np.linalg.norm(psd_coords[:, None, :] - spine_bases[None, :, :], axis=2)
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            psd_idx, spine_idx = min_idx

            matches.append({
                'id': ids[psd_idx],
                'pre_root_id': pre_root_ids[psd_idx],
                'activity_level': mapped_activity_levels[psd_idx],
                'post_root_id': cellid,
                'psd_coords': psd_coords[psd_idx],
                'spine_coords': spine_bases[spine_idx]
            })

            filtered_psds[str(cellid)].append([psd_coords[psd_idx].tolist(), mapped_activity_levels[psd_idx]])

            snapped_points.append(spine_bases[spine_idx] * 1000)
            microns_points.append(psd_coords[psd_idx] * 1000)

            # Remove matched points
            psd_coords = np.delete(psd_coords, psd_idx, axis=0)
            mapped_activity_levels = np.delete(mapped_activity_levels, psd_idx, axis=0)
            ids = np.delete(ids, psd_idx, axis=0)
            pre_root_ids = np.delete(pre_root_ids, psd_idx, axis=0)
            spine_bases = np.delete(spine_bases, spine_idx, axis=0)
    
    #print(pd.DataFrame.from_dict(matches))
    with open("../data/jsons/filtered_psds.json", "w") as outfile: 
        json.dump(filtered_psds, outfile)


        if snapped_points:
            client = CAVEclient('pinky_sandbox')
            cell_source = "precomputed://https://storage.googleapis.com/microns_public_datasets/pinky100_v185/seg"
            mesh_dir = '../MeshCache/'
            mm = trimesh_io.MeshMeta(cv_path=cell_source, disk_cache_path=mesh_dir)
            cell_mesh = mm.mesh(seg_id=cellid, remove_duplicate_vertices=True)
            cell_mesh.add_link_edges(seg_id=cellid, client=client.chunkedgraph)

            comp_mask = mesh_filters.filter_largest_component(cell_mesh)
            mask_filt = cell_mesh.apply_mask(comp_mask)

            #stuff for long term solution

            


            cell_mesh_actor = trimesh_vtk.mesh_actor(mask_filt, opacity=0.3, color=(0.7, 0.7, 0.7))
            snapped_actor = trimesh_vtk.point_cloud_actor(snapped_points, size=30, color=(0, 1, 0), opacity=1)
            microns_actor = trimesh_vtk.point_cloud_actor(microns_points, size=30, color=(0, 0, 1), opacity=1)
            red_line_actor = trimesh_vtk.linked_point_actor(vertices_a=np.array(snapped_points), vertices_b=np.array(microns_points), color=(1, 0, 0), opacity=1)
            
            # Render all actors, including the red lines
            camera = trimesh_vtk.oriented_camera(cell_mesh.centroid, backoff=400)
            #trimesh_vtk.render_actors([cell_mesh_actor, snapped_actor, microns_actor, red_line_actor], camera=camera)
