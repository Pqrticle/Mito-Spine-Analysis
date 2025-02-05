from meshparty import trimesh_io, mesh_filters
from caveclient import CAVEclient
import trimesh
import subprocess
import pandas as pd
import json
import os
import random
import activity_clustering
import Guillotine
import coordinate_matching
import mito_detection

'''
request = int(input('Select the type of analysis: '))
if request == 0:
    pass
    #neurons_list = 
    # 
    #   random.seed(42)
    low_activity_levels, high_activity_levels = activity_clustering.activity_levels()
    low_activity_neurons = random.sample(list(low_activity_levels), 86)
    high_activity_neurons = random.sample(list(high_activity_levels), 22)
    selected_neuron_ids = low_activity_neurons + high_activity_neurons'''

mark_bases = True
run_blender = False
pipline = True
detect_mito = True
activity = True


if __name__ == '__main__':
    raw_mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
    neuron_seg_source = "precomputed://https://storage.googleapis.com/microns_public_datasets/pinky100_v185/seg"
    mito_seg_source = "precomputed://https://td.princeton.edu/sseung-archive/pinky100-mito/seg_191220"
    neuron_mesh_dir = '../data/meshes/neuron_meshes/'  # Neurons preloaded locally
    mito_mesh_dir = '../data/meshes/mito_meshes/'  # Mito downloaded through cloud-volume
    
    mm = trimesh_io.MeshMeta(cv_path=neuron_seg_source, disk_cache_path=neuron_mesh_dir, cache_size=20)
    mito_mm = trimesh_io.MeshMeta(cv_path=mito_seg_source, disk_cache_path=mito_mesh_dir, cache_size=20)
    
    client = CAVEclient()
    auth = client.auth
    client = CAVEclient('pinky_sandbox')

    num = 0
    neurons_list = [648518346349538239]
    for neuron_id in neurons_list:
        neuron_id = int(neuron_id)
        print(f'Processing Neuron ID: {neuron_id}')
        num += 1
        try:
            neuron_mesh = mm.mesh(seg_id=neuron_id, remove_duplicate_vertices=True)
            neuron_mesh.add_link_edges(seg_id=neuron_id, client=client.chunkedgraph)
            print(f"Neuron mesh loaded: {neuron_mesh.n_vertices} vertices, {neuron_mesh.n_faces} faces")
            if mark_bases and f"{neuron_id}-SBC.json" not in os.listdir('../data/spine_base_coords'):
                comp_mask = mesh_filters.filter_largest_component(neuron_mesh) #quotes at the start of this line
                mask_filter = neuron_mesh.apply_mask(comp_mask)
                Guillotine.snapped_spine_base_coords(mask_filter, neuron_id) #quotes here
        except ValueError:
            print(f'No mesh could be found for Neuron SegID: {neuron_id}')

        if pipline:
            status = coordinate_matching.match_coords(neuron_id)
            if status is None:
                detect_mito = False
                print(f'Neuron {neuron_id} not found in filtered synapse table')


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

        # Combine and name all meshes within a scene to be saved as separate objects in the OBJ file
        scene = trimesh.Scene()
        scene.add_geometry(neuron_mesh, geom_name=str(neuron_id))
        for mito_id, mito_mesh in mito_meshes.items():
            scene.add_geometry(mito_mesh, geom_name=str(mito_id))

        # Exports the OBJ file
        neuron_obj_path = f'../data/meshes/final_meshes/{neuron_id}-MSH.obj'
        scene.export(neuron_obj_path)
        print(f"Saved combined mesh to {neuron_obj_path}")

        if detect_mito:
            mito_detection.detect_nearby_mito(neuron_id, mito_meshes)

        if activity:
            activity_clustering.sum_spike_probability(neuron_id)

        # Run Blender with the provided Python script
        if run_blender:
            subprocess.run(["cmd", "/c", r"C:\Program Files\Blender-2.93-CellBlender\blender.exe", "--python", r"C:\Users\PishosL\Mito-Spine-Analysis\blender\mito_blender.py", 
                    "--background", 
                    "--", str(neuron_id)])
    print(f'Number of Neurons Completed (current analysis): {num}')

    
