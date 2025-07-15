import guillotine
import mito_metrics
import selectivity_index
from meshparty import trimesh_io
from caveclient import CAVEclient

def process_neurons(radius):
    neuron_seg_source = "precomputed://https://storage.googleapis.com/microns_public_datasets/pinky100_v185/seg"
    mito_seg_source = "precomputed://https://td.princeton.edu/sseung-archive/pinky100-mito/seg_191220"
    neuron_mesh_dir = '../data/meshes/neuron_meshes/'  # Neurons preloaded locally
    mito_mesh_dir = '../data/meshes/mito_meshes/'  # Mito downloaded through cloud-volume
    
    mm = trimesh_io.MeshMeta(cv_path=neuron_seg_source, disk_cache_path=neuron_mesh_dir, cache_size=20)
    mito_mm = trimesh_io.MeshMeta(cv_path=mito_seg_source, disk_cache_path=mito_mesh_dir, cache_size=20)
    client = CAVEclient('pinky_sandbox')

    count = 1
    selectivity_indexes = selectivity_index.get_selectivity_indexes()
    for neuron_id, (osi, dsi) in selectivity_indexes.items():
        # Retrieve mesh and find spine base coordinates
        print('\n')
        print(f'Neuron ID: {neuron_id} (#{count}/{len(selectivity_indexes)})')
        neuron_mesh = mm.mesh(seg_id=neuron_id, remove_duplicate_vertices=True)
        neuron_mesh.add_link_edges(seg_id=neuron_id, client=client.chunkedgraph)
        neuron_mesh.vertices /= 1000
        print('Loaded neuron mesh')
        guillotine.snap_spine_bases(neuron_id, neuron_mesh)
    
        #mito stuff
        mito_meshes = mito_metrics.generate_mito_meshes(neuron_id, mito_mm, mito_mesh_dir)
        mito_metrics.detect_nearby_mito(neuron_id, neuron_mesh, mito_meshes, osi, dsi, radius)
        count += 1

if __name__ == '__main__':
    radius=1
    process_neurons(radius)