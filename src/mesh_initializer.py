import guillotine
import mito_metrics
import ast
import pandas as pd
from meshparty import trimesh_io
from caveclient import CAVEclient

def process_neurons(radius):
    osi_data = pd.read_csv("../data/osi_data.csv")
    neuron_seg_source = "precomputed://https://storage.googleapis.com/microns_public_datasets/pinky100_v185/seg"
    mito_seg_source = "precomputed://https://td.princeton.edu/sseung-archive/pinky100-mito/seg_191220"
    neuron_mesh_dir = '../data/meshes/neuron_meshes/'  # Neurons preloaded locally
    mito_mesh_dir = '../data/meshes/mito_meshes/'  # Mito downloaded through cloud-volume
    
    mm = trimesh_io.MeshMeta(cv_path=neuron_seg_source, disk_cache_path=neuron_mesh_dir, cache_size=20)
    mito_mm = trimesh_io.MeshMeta(cv_path=mito_seg_source, disk_cache_path=mito_mesh_dir, cache_size=20)
    client = CAVEclient('pinky_sandbox')

    osi_data["osi"] = osi_data["osi"].apply(lambda x: float(ast.literal_eval(x)[0]))
    count = 1
    for idx, row in osi_data.iterrows():
        neuron_id = row["neuron_id"]
        osi = row["osi"]
        if osi > 0.6 or osi < 0.06:
            print(neuron_id, count)
            try:
                neuron_mesh = mm.mesh(seg_id=neuron_id, remove_duplicate_vertices=True)
                neuron_mesh.add_link_edges(seg_id=neuron_id, client=client.chunkedgraph)
                neuron_mesh.vertices /= 1000
                print(f"Neuron mesh loaded: {neuron_mesh.n_vertices} vertices, {neuron_mesh.n_faces} faces")
                guillotine.snap_spine_bases(neuron_id, neuron_mesh)
            except ValueError:
                print(f'No mesh could be found for Neuron SegID: {neuron_id}')

            mito_meshes = mito_metrics.generate_mito_meshes(neuron_id, mito_mm, mito_mesh_dir)
            mito_metrics.detect_nearby_mito(neuron_id, neuron_mesh, mito_meshes, osi, radius)
            count += 1

if __name__ == '__main__':
    radius=1.5
    process_neurons(radius)