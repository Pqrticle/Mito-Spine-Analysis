# this script requires some weird version of python 3.9 and bpy to run from the terminal or within the project
# instead this script is loaded into CellBlender and ran from within
# ignore the errors/warnings this file may indicate, they're just growing pains!
# i installed pandas 3.9 in blender

import bpy
import json
import sys

neuron_id = int(sys.argv[-1])
obj_file_path = f'../data/meshes/final_meshes/{neuron_id}-MSH.obj'  # Replace with your .obj file path
blend_file_path = f'../data/meshes/final_blends/{neuron_id}-MSH.blend' # Replace with your desired .blend file path

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

bpy.ops.object.select_all(action='DESELECT')  # Select all objects
neuron_obj = bpy.data.objects.get(str(neuron_id))
neuron_obj.name = 'Neuron_Mesh'

bpy.ops.object.select_all(action='SELECT')  # Select all objects
neuron_obj.select_set(False)

combined_mito_obj = bpy.data.objects[0]
bpy.context.view_layer.objects.active = combined_mito_obj
bpy.ops.object.join()  # Combine mitochondria into one object
combined_mito_obj.name = 'Combined_Mito_Mesh'
bpy.ops.object.select_all(action='DESELECT')

with open("../data/jsons/filtered_psds.json", "r") as file: 
        filtered_psds = json.load(file)

num = 1
psd_supersets = filtered_psds[str(neuron_id)]
for psd_set in psd_supersets:
    psd, activity_level = psd_set
    psd_x, psd_y, psd_z = psd[0], psd[1], psd[2]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=10, location=(psd_x, -psd_z, psd_y))
    sphere_obj = bpy.context.object
    sphere_obj.name = f"Sphere-{num}"
    bpy.ops.object.select_all(action='DESELECT')
    print(1)
    combined_mito_obj.select_set(True)
    bpy.ops.object.duplicate()  # Duplicate the selected mitochondrion
    new_mito_obj = bpy.data.objects.get(f'Combined_Mito_Mesh.001')
    combined_mito_obj.select_set(False)
    new_mito_obj.select_set(False)

    neuron_obj.select_set(True)
    bpy.ops.object.duplicate()  # Duplicate the selected mitochondrion
    new_neuron_obj = bpy.data.objects.get(f'Neuron_Mesh.001')
    neuron_obj.select_set(False)
    
    new_mito_obj.select_set(True)
    new_neuron_obj.select_set(True)

    combined_neuron_mito_obj = new_neuron_obj
    bpy.context.view_layer.objects.active = combined_neuron_mito_obj
    bpy.ops.object.join()  # Combine mitochondria into one object
    combined_neuron_mito_obj.name = 'Combined_Neuron_Mito_Mesh'
    bpy.ops.object.select_all(action='DESELECT')

    # Apply Boolean modifier
    bool_mod = combined_neuron_mito_obj.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = 'INTERSECT'
    bool_mod.object = sphere_obj
    bpy.ops.object.modifier_apply(modifier="Boolean")

    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    sphere_obj.select_set(True)  # Select the sphere
    sphere_obj.scale = (0.1, 0.1, 0.1)
    bpy.context.view_layer.objects.active = sphere_obj
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = combined_mito_obj
    new_material = bpy.data.materials.new('Material.001')
    combined_mito_obj.data.materials.append(new_material)
    combined_mito_obj.data.materials.pop(index=0)

    bpy.ops.object.select_all(action='DESELECT')
    neuron_obj.active_material.diffuse_color[3] = 0.0666656

    # Save the blender file
    print(activity_level, neuron_id, num)
    bpy.ops.wm.save_as_mainfile(filepath=f'Z:/DATA/Luca/Mitochondrial Correlates of Synaptic Plasticity/psd_intersections/{activity_level}/{neuron_id}-{num}-MSH.blend')
    print("Imported and saved the .blend file successfully.")

    sphere_obj.hide_set(True)
    bpy.data.objects.remove(combined_neuron_mito_obj, do_unlink=True)
    num += 1