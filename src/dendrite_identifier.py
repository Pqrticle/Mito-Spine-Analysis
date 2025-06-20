import json
import numpy as np
import trimesh
import pandas as pd
from trimesh.boolean import intersection
from scipy.spatial import cKDTree
from collections import defaultdict

def resolve_duplicate_nonmanifold_edges(mesh, tol=1e-2):
    """
    Resolves geometrically similar non-manifold edges by collapsing them to a single canonical version.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to clean.

    tol : float
        The distance tolerance for grouping edges (in mesh units).

    Returns
    -------
    cleaned : trimesh.Trimesh
        The mesh with resolved edge redundancy and updated face connectivity.
    """
    faces = mesh.faces.copy()
    vertices = mesh.vertices.copy()

    # Step 1: Identify non-manifold edges
    edges_sorted = mesh.edges_sorted.reshape(-1, 2)
    edge_hash = trimesh.grouping.hashable_rows(edges_sorted)
    edges_unique, edge_inverse, edge_counts = np.unique(edge_hash, return_inverse=True, return_counts=True)
    nonmanifold_mask = edge_counts != 2
    nonmanifold_edges = mesh.edges_unique[nonmanifold_mask]

    if len(nonmanifold_edges) == 0:
        print("No non-manifold edges found.")
        return mesh.copy()

    # Step 2: Compute edge midpoints
    edge_coords = np.array([[vertices[i], vertices[j]] for i, j in nonmanifold_edges])
    edge_mids = edge_coords.mean(axis=1)

    # Step 3: Build KDTree to find close edge midpoints
    tree = cKDTree(edge_mids)
    neighbors = tree.query_ball_tree(tree, tol)

    # Step 4: Collapse overlapping groups using union-find
    parent = list(range(len(neighbors)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, group in enumerate(neighbors):
        for j in group:
            union(i, j)

    groups = defaultdict(list)
    for idx in range(len(parent)):
        groups[find(idx)].append(idx)

    print(f"Found {len(groups)} groups of redundant non-manifold edges (within {tol} nm):")
    for k, g in groups.items():
        if len(g) > 1:
            print(f"  Group {k}: edges {g}")

    # Step 5: Create vertex remap dictionary
    edge_to_vertex = {}
    for group in groups.values():
        if len(group) <= 1:
            continue
        all_vertices = np.unique(nonmanifold_edges[group].flatten())
        ref_vertex = np.min(all_vertices)
        for v in all_vertices:
            if v != ref_vertex:
                edge_to_vertex[v] = ref_vertex

    # Step 6: Update face connectivity
    for old_v, new_v in edge_to_vertex.items():
        faces[faces == old_v] = new_v

    # Step 7: Clean up mesh
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    new_mesh.update_faces(new_mesh.nondegenerate_faces())
    new_mesh.remove_unreferenced_vertices()

    return new_mesh

def calculate_dendritic_volume(neuron_id, neuron_mesh):
    synapse_table = pd.read_csv('../data/synapse_table.csv')
    synapse_df = synapse_table[synapse_table['post_root_id'] == neuron_id]
    spine_bases = synapse_df[synapse_df['distance'] < 1]['base_coords'].apply(json.loads).tolist() #distnace must be less than 1um

    sphere_meshes = []
    for base_coord in spine_bases:
        base_coord = np.array(base_coord)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=3)
        sphere.apply_translation(base_coord)
        sphere_meshes.append(sphere)
    merged_sphere = trimesh.boolean.union(sphere_meshes, engine='manifold')
    
    face_areas = merged_sphere.area_faces
    valid_faces_mask = face_areas > 0
    cleaned_sphere = merged_sphere.submesh([valid_faces_mask], append=True, repair=False)
    
    final_sphere_mesh = resolve_duplicate_nonmanifold_edges(cleaned_sphere, tol=1e-1)
    dendrite_mesh_raw = trimesh.boolean.intersection([neuron_mesh, final_sphere_mesh], engine='manifold')

    parts = dendrite_mesh_raw.split(only_watertight=False)

    filtered_parts = []
    for p in parts:
        sphericity = (np.pi ** (1 / 3)) * ((6 * p.volume) ** (2 / 3)) / p.area
        if sphericity < 0.7 or p.volume < 5:
            filtered_parts.append(p)
        print(sphericity, p.volume)

    print(len(parts))
    print(len(filtered_parts))
    dendrite_mesh = trimesh.util.concatenate(filtered_parts)

    scene = trimesh.Scene()
    scene.add_geometry(neuron_mesh, node_name='neuron_mesh')
    scene.add_geometry(final_sphere_mesh, node_name='sphere_mesh')
    scene.add_geometry(dendrite_mesh, node_name='dendrite_mesh_split')
    scene.export(f"../data/meshes/dendrite_meshes/{neuron_id}-DEN.obj")

    # Print volumes
    print(f"Neuron volume: {neuron_mesh.volume:.3f} μm³")
    print(f"Dendritic raw volume: {dendrite_mesh_raw.volume:.3f} μm³")
    print(f"Dendritic final volume: {dendrite_mesh.volume:.3f} μm³")

    return dendrite_mesh.volume