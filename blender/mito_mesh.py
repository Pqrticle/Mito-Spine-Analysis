from meshparty import trimesh_io, mesh_filters
from caveclient import CAVEclient
import trimesh
import subprocess
import pandas as pd
import json
import os
import sys
import random
import activity_clustering
import Guillotine
import mito_graphs
import matplotlib.pyplot as plt
import seaborn as sns

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

    low_activity_levels, high_activity_levels = activity_clustering.activity_levels()

    random.seed(42)
    low_activity_neurons = random.sample(list(low_activity_levels), 15)
    high_activity_neurons = random.sample(list(high_activity_levels), 15)

    print(low_activity_neurons)
    print(high_activity_neurons)

    selected_neuron_ids = low_activity_neurons + high_activity_neurons
    print(low_activity_neurons)
    print(high_activity_neurons)

    with open("../data/jsons/test-3.json", "r") as file: 
        data = json.load(file)
    

    for neuron_id in selected_neuron_ids:
        if str(neuron_id) not in data:
            print(f'Processing Neuron ID: {neuron_id}')
            '''try:
                neuron_mesh = mm.mesh(seg_id=neuron_id, remove_duplicate_vertices=True)
                neuron_mesh.add_link_edges(seg_id=neuron_id, client=client.chunkedgraph)
                comp_mask = mesh_filters.filter_largest_component(neuron_mesh)
                mask_filter = neuron_mesh.apply_mask(comp_mask)
                print(f"Neuron mesh loaded: {neuron_mesh.n_vertices} vertices, {neuron_mesh.n_faces} faces")
            except ValueError:
                print(f'No mesh could be found for Neuron SegID: {neuron_id}')

            # Get spine coordinates
            spine_base_coords = Guillotine.snapped_spine_base_coords(mask_filter)
            with open(f'../data/jsons/{neuron_id}-spine-base-coords', 'w') as file:
                json.dump(spine_base_coords, file)

            # Filter for mito within dendrites of the neuron
            all_mito_df = raw_mito_data[raw_mito_data.cellid == neuron_id]
            dendritic_mito_df = all_mito_df[all_mito_df.compartment.isin(['Basal', 'Apical'])]
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
            print(f"Saved combined mesh with separate objects to {neuron_obj_path}")'''
            
            # Run Blender with the provided Python script
            subprocess.run(["cmd", "/c", r"C:\Program Files\Blender-2.93-CellBlender\blender.exe", "--python", r"C:\Users\PishosL\Mito-Spine-Analysis\blender\mito_blender.py", "--background", "--", str(neuron_id)])

    mito_graphs.graph(high_activity_neurons)

    
        # add "--background" eventually
    #mito_graphs.plot2(selected_neuron_ids)


    #make sure you are cd into blender directory
    #cd "C:\Users\PishosL\EM-Cristae-Detection\blender\"
    #"C:\Program Files\Blender-2.93-CellBlender\blender.exe" --python "C:\Users\PishosL\EM-Cristae-Detection\blender\mito_blender.py" -- "981253"