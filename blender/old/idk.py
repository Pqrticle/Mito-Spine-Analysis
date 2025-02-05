import pandas as pd
import numpy as np
import json
from meshparty import trimesh_io, mesh_filters, trimesh_vtk
from caveclient import CAVEclient
import Guillotine
import skeletor as sk

if __name__ == '__main__':
    raw_spine_data = pd.read_csv('../data/pni_synapses_v185.csv')
    raw_mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
    unique_cellids = raw_mito_data['cellid'].unique()
   
    for cellid in unique_cellids:
        neuron_seg_source = "precomputed://https://storage.googleapis.com/microns_public_datasets/pinky100_v185/seg"
        mito_seg_source = "precomputed://https://td.princeton.edu/sseung-archive/pinky100-mito/seg_191220"
        neuron_mesh_dir = '../data/meshes/neuron_meshes/'  # Neurons preloaded locally
        mito_mesh_dir = '../data/meshes/mito_meshes/'  # Mito downloaded through cloud-volume
              
        client = CAVEclient()
        auth = client.auth
        client = CAVEclient('pinky_sandbox')

        mm = trimesh_io.MeshMeta(cv_path=neuron_seg_source, disk_cache_path=neuron_mesh_dir, cache_size=20)
        mito_mm = trimesh_io.MeshMeta(cv_path=mito_seg_source, disk_cache_path=mito_mesh_dir, cache_size=20)
        cell_mesh = mm.mesh(seg_id=cellid, remove_duplicate_vertices=True)
        cell_mesh.add_link_edges(seg_id=cellid, client=client.chunkedgraph)

        #Filter for mito within dendrites of the neuron
        all_mito_df = raw_mito_data[raw_mito_data.cellid == cellid]
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


        comp_mask = mesh_filters.filter_largest_component(cell_mesh)
        mask_filt = cell_mesh.apply_mask(comp_mask)

        skel = sk.skeletonize.by_wavefront(mask_filt, origins=None, waves=1, step_size=1)
        sk.post.remove_bristles(skel, los_only=False, inplace=True)
        sk.post.clean_up(skel, inplace=True, theta=1)

        polylines, radii = Guillotine.get_branch_polylines_by_length(skel, min_length=30000, max_length=99999999)

            
        #something about finding the center of vectors dude i dont know


        cell_mesh_actor = trimesh_vtk.mesh_actor(mask_filt, opacity=0.3, color=(0.7, 0.7, 0.7))
        
        
        # Render all actors, including the red lines
        camera = trimesh_vtk.oriented_camera(cell_mesh.centroid, backoff=400)
        #trimesh_vtk.render_actors([cell_mesh_actor, snapped_actor, microns_actor, red_line_actor], camera=camera)