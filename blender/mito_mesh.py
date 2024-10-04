from meshparty import trimesh_io
import trimesh
import subprocess
import pandas as pd
import os
import sys
import random
import mito_graphs

if __name__ == '__main__': # prevents multithreading loop
    raw_mito_data = pd.read_csv('../data/pni_mito_analysisids_fullstats.csv')
    neuron_seg_source = "precomputed://https://storage.googleapis.com/microns_public_datasets/pinky100_v185/seg"
    mito_seg_source = "precomputed://https://td.princeton.edu/sseung-archive/pinky100-mito/seg_191220"
    neuron_mesh_dir = '../data/meshes/neuron_meshes/' # neurons preloaded locally
    mito_mesh_dir = '../data/meshes/mito_meshes/' # mito downloaded through cloud-volume
    mm = trimesh_io.MeshMeta(cv_path=neuron_seg_source, disk_cache_path=neuron_mesh_dir, cache_size=20)
    mito_mm = trimesh_io.MeshMeta(cv_path=mito_seg_source, disk_cache_path=mito_mesh_dir, cache_size=20)

    unique_neuron_ids = raw_mito_data.cellid.drop_duplicates()
    selected_neuron_ids = unique_neuron_ids.sample(n=3, random_state=1).tolist()
    print(selected_neuron_ids)
    for neuron_id in selected_neuron_ids:
        #try:
            #neuron_id = int(sys.argv[1])
        #except IndexError:
            #neuron_id = 648518346349492197
            #print(f'No Neuron SegID was provided, defaulting to: {neuron_id}')

        print(f'Neuron ID: {neuron_id}')

        try:
            neuron_mesh = mm.mesh(seg_id=neuron_id, remove_duplicate_vertices=True)
            print(f"Neuron mesh loaded: {neuron_mesh.n_vertices} vertices, {neuron_mesh.n_faces} faces")
        except ValueError:
            print(f'No mesh could be found for Neuron SegID: {neuron_id}')
            sys.exit()

        # filter for mito within dendrites of the neuron
        all_mito_df = raw_mito_data[raw_mito_data.cellid == neuron_id]
        dendritic_mito_df = all_mito_df[all_mito_df.compartment.isin(['Basal', 'Apical'])]
        mito_ids = dendritic_mito_df.mito_id.tolist()

        # determine if mito are already cached
        mito_meshes = {}
        for mito_id in mito_ids:
            mito_id_path = f'{mito_mesh_dir}/{mito_id}.h5'
            if not os.path.exists(mito_id_path):
                mito_mesh = mito_mm.mesh(seg_id=mito_id, remove_duplicate_vertices=True)
                print(f"Loaded mitochondria mesh: {mito_id}")
            else:
                mito_mesh = mito_mm.mesh(filename=mito_id_path)
            mito_meshes[mito_id] = mito_mesh

        '''# combine and name all meshes within a scene to be saved as separate objects in the OBJ file
        random_mito_id, random_mito_mesh = random.choice(list(mito_meshes.items()))
        scene = trimesh.Scene()
        scene.add_geometry(neuron_mesh, geom_name=str(neuron_id))
        scene.add_geometry(random_mito_mesh, geom_name=str(random_mito_id))
        print(f'Mito: {random_mito_id}')
        '''
        # combine and name all meshes within a scene to be saved as separate objects in the OBJ file
        scene = trimesh.Scene()
        scene.add_geometry(neuron_mesh, geom_name=str(neuron_id))
        for mito_id, mito_mesh in mito_meshes.items():
            scene.add_geometry(mito_mesh, geom_name=str(mito_id))

        # exports the OBJ file
        neuron_obj_path = f'../data/meshes/final_meshes/{neuron_id}-MSH.obj'
        scene.export(neuron_obj_path)
        print(f"Saved combined mesh with separate objects to {neuron_obj_path}")
        
        subprocess.run(["cmd", "/c", r"C:\Program Files\Blender-2.93-CellBlender\blender.exe", "--python", r"C:\Users\PishosL\EM-Cristae-Detection\blender\mito_blender.py", "--background", "--", str(neuron_id)])

        # add "--background" eventually
    mito_graphs.plot2(selected_neuron_ids)


    #make sure you are cd into blender directory
    #cd "C:\Users\PishosL\EM-Cristae-Detection\blender\"
    #"C:\Program Files\Blender-2.93-CellBlender\blender.exe" --python "C:\Users\PishosL\EM-Cristae-Detection\blender\mito_blender.py" -- "981253"