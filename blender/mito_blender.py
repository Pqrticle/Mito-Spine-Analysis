# this script requires some weird version of python 3.9 and bpy to run from the terminal or within the project
# instead this script is loaded into CellBlender and ran from within
# ignore the errors/warnings this file may indicate, they're just growing pains!
# i installed pandas 3.9 in blender

import bpy
import bmesh
import sys
import math
import json
import time
import random
sys.path.append(r"C:\Users\PishosL\Mito-Spine-Analysis\blender")
import mito_stats

def calculate_volume(obj):
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    volume = bm.calc_volume()
    bm.free()
    return volume

def is_partially_within_sphere(obj, sphere_center, radius):
    for vertex in obj.data.vertices:
        distance = math.sqrt((sphere_center[0]-vertex.co[0])**2 + (sphere_center[1]-vertex.co[1])**2 + (sphere_center[2]-vertex.co[2])**2)
        if distance <= radius:
            return True

neuron_id = int(sys.argv[-1])
obj_file_path = f'../data/meshes/final_meshes/{neuron_id}-MSH.obj'  # Replace with your .obj file path
blend_file_path = f'../data/meshes/final_blends/{neuron_id}-MSH.blend' # Replace with your desired .blend file path

with open(f'../data/jsons/{neuron_id}-spine-base-coords', 'r') as file:
        spine_base_coords = json.load(file)

# Import neuron and mitochondria meshes
bpy.ops.import_scene.obj(filepath=obj_file_path)
scene = bpy.context.scene
bpy.data.objects.remove(bpy.data.objects.get('Camera'))
bpy.data.objects.remove(bpy.data.objects.get('Cube'))
bpy.data.objects.remove(bpy.data.objects.get('Light'))

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
print('arena scaled')

# Apply scale to all objects
scaling_factor = (0.001, 0.001, 0.001)
bpy.ops.object.select_all(action='SELECT')  # Select all objects
bpy.ops.transform.resize(value=scaling_factor)  # Resize all selected objects
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # Apply scaling

neuron_obj = bpy.data.objects.get(str(neuron_id))
neuron_obj.select_set(False)

combined_mito_obj = bpy.data.objects[0]
bpy.context.view_layer.objects.active = combined_mito_obj
bpy.ops.object.join()  # Combine mitochondria into one object
bpy.ops.object.select_all(action='DESELECT')
combined_mito_obj.name = 'Combined_Mito_Mesh'

#for mito_id in mito_list:
'''mito_obj = bpy.data.objects.get(str(mito_id))
    bpy.context.view_layer.objects.active = mito_obj

    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels = 3
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    bpy.ops.object.modifier_apply(modifier="Triangulate")

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.gamer.normal_smooth()

    # Calculate the mesh curvatures
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.gamer.compute_curvatures()
    curvatures_list = mito_obj.gamer.curvatures.curvature_list

    # Retrieve K1 curvature values
    mito_mesh = mito_obj.data
    K1_curvature = curvatures_list[0]
    k1_layer = mito_mesh.vertex_layers_float['MDSBK1']
    k1_values = [k1_layer.data[vertex.index].value for vertex in mito_mesh.vertices]

    # Retrieve the XYZ coordinates of each vertex
    vertex_coordinates = [vertex.co for vertex in mito_mesh.vertices]
    curvature_and_coords = [[float(k1_values[i]), float(vertex_coordinates[i].x), float(vertex_coordinates[i].y), float(vertex_coordinates[i].z)] for i in range(len(vertex_coordinates))]

    spine_x, spine_y, spine_z = mito_stats.nearest_spine_coords(neuron_id, mito_id)
    mito_stats.curves_within_sphere(curvature_and_coords, spine_x, spine_y, spine_z)'''

num = 1
intersected_mito_volumes = []
for base_coords in spine_base_coords:
    # makes temp sphere
    spine_x, spine_y, spine_z = base_coords[0], base_coords[1], base_coords[2]
    #print(f'Sphere-{num}')
    #print(spine_x, spine_y, spine_z)
    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(spine_x, -spine_z, spine_y))
    sphere_obj = bpy.context.object
    sphere_obj.name = f"Sphere-{num}"
    sphere_center = (spine_x, spine_y, spine_z)
    
    bpy.ops.object.select_all(action='DESELECT')
    num += 1

    if is_partially_within_sphere(combined_mito_obj, sphere_center, 1):
        combined_mito_obj.select_set(True)
        bpy.ops.object.duplicate()  # Duplicate the selected mitochondrion
        intersected_mito_obj = bpy.data.objects.get('Combined_Mito_Mesh.001')
        #print(intersected_mito_obj.name)

        # Apply Boolean modifier
        bool_mod = intersected_mito_obj.modifiers.new(name="Boolean", type='BOOLEAN')
        bool_mod.operation = 'INTERSECT'
        bool_mod.object = sphere_obj

        # Apply the Boolean modifier
        bpy.ops.object.modifier_apply(modifier="Boolean")
        intersected_mito_volume = calculate_volume(intersected_mito_obj)

        # Optionally, you can remove the original combined object if needed
        bpy.data.objects.remove(intersected_mito_obj, do_unlink=True)
        #print(base_coords, intersected_mito_volume)
    else:
        intersected_mito_volume = 0
    intersected_mito_volumes.append(intersected_mito_volume)
    print(f'{sphere_obj.name} Volume: {intersected_mito_volume}')
    time.sleep(2)


json_dict = {neuron_id: intersected_mito_volumes}

try:
    with open("../data/jsons/test-3.json", "r") as file: 
        data = json.load(file)
    data.update(json_dict)
    json_dict = data
except:
    pass

# Write the updated data back to the JSON file
with open("../data/jsons/test-3.json", "w") as file:
    json.dump(json_dict, file, indent=4)  # U


# Save the blender file
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)
print("Imported and saved the .blend file successfully.")