# this script requires some weird version of python 3.9 and bpy to run from the terminal or within the project
# instead this script is loaded into CellBlender and ran from within
# ignore the errors/warnings this file may indicate, they're just growing pains!
# i installed pandas 3.9 in blender

import bpy
import sys
import json
import pandas as pd
sys.path.append(r"C:\Users\PishosL\EM-Cristae-Detection\blender")
import mito_stats

neuron_id = int(sys.argv[-1])
obj_file_path = f'../data/meshes/final_meshes/{neuron_id}-MSH.obj'  # Replace with your .obj file path
blend_file_path = f'../data/meshes/final_blends/{neuron_id}-MSH.blend' # Replace with your desired .blend file path

# Import neuron and mitochondria meshes
bpy.ops.import_scene.obj(filepath=obj_file_path)
scene = bpy.context.scene

# Set the field of view
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        # Get the space data for the 3D Viewport
        space = area.spaces.active
        if space.type == 'VIEW_3D':
            # Change Clip Start and Clip End
            space.clip_start = 0.1
            space.clip_end = 1000
            
            if space.overlay:
                space.overlay.grid_scale = 1e-06  # Set a small grid scale
override = {'area': area, 'region': area.regions[-1]}

# Resize the neuron and mitochondrial meshes to the correct micrometer values
# Note that the units of operation in Blender are meters
mito_list = []
scaling_tuple = (0.001, 0.001, 0.001)
for obj in bpy.data.objects:
    try:
        obj_name = int(obj.name)
        mesh_obj = bpy.data.objects.get(obj.name)
        mesh_obj.scale = scaling_tuple
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.ops.view3d.view_selected(override)
        if obj_name != neuron_id:
            mito_list.append(obj_name)
    except:
        pass

#SET UP DUPLICATE FOR NEURON
# Duplicate the original mitochondria mesh
#mito_obj = bpy.data.objects.get(str(mito_id))
#bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(-0, -15, -0)})
#bpy.ops.object.select_all(action='DESELECT')

mito_ratios = {}
for mito_id in mito_list:
    mito_obj = bpy.data.objects.get(str(mito_id))
    bpy.context.view_layer.objects.active = mito_obj

    '''bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels = 3
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    bpy.ops.object.modifier_apply(modifier="Triangulate")

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.gamer.normal_smooth()'''

    # Calculate the mesh curvatures
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.gamer.compute_curvatures()
    curvatures_list = mito_obj.gamer.curvatures.curvature_list

    # Retrieve K1 curvature values
    mito_mesh = mito_obj.data
    K1_curvature = curvatures_list[0]
    k1_layer = mito_mesh.vertex_layers_float['MDSBK1']
    k1_values = [k1_layer.data[vertex.index].value for vertex in mito_mesh.vertices]
    #print(k1_values)

    # Retrieve the XYZ coordinates of each vertex
    vertex_coordinates = [vertex.co for vertex in mito_mesh.vertices]
    curvature_and_coords = [[float(k1_values[i]), float(vertex_coordinates[i].x), float(vertex_coordinates[i].y), float(vertex_coordinates[i].z)] for i in range(len(vertex_coordinates))]

    # Write curvature and coordinate information into a JSON
    with open(f'../data/jsons/{neuron_id}-{mito_id}-curvature_data.json', 'w') as f:
        json.dump(curvature_and_coords, f)

    #makes temp sphere
    spine_x, spine_y, spine_z = mito_stats.nearest_spine_coords(neuron_id, mito_id)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=2, location=(spine_x, -spine_z, spine_y))
    sphere = bpy.context.object
    sphere.name = f"Sphere-{mito_id}"

    ratio, mito_list = mito_stats.vertex_curvatures_ratio(neuron_id, mito_id)
    mito_ratios[mito_id] = (ratio, k1_values)

df = pd.DataFrame([(mito_id, ratio, avg_curvature) for mito_id, (ratio, avg_curvature) in mito_ratios.items()], columns=['Mito', 'Ratio', 'Curvatures'])
df.to_csv(f'../data/csvs/{neuron_id}-mito_ratios.csv', index=False)

mito_stats.curves_within_range(neuron_id, mito_id, spine_x, spine_y, spine_z)
print(mito_list)

# Save the blender file
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)
print("Imported and saved the .blend file successfully.")