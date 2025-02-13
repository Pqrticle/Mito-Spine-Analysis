import pickle
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

with open("../data/Stimulus.pkl", "rb") as f:
    Stimulus = pickle.load(f)

with open("../data/EASETrace.pkl", "rb") as f:
    EASETrace = pickle.load(f)

with open("../data/EASETuning.pkl", "rb") as f:
    EASETuning = pickle.load(f)

def get_section(conditions, angle):
  if np.isnan(angle):
    valid = np.isnan(conditions).astype("int")
  else:
    valid = (conditions==angle).astype("int")
    
  valid_diff = np.diff(valid)
  valid_diff = valid_diff.reshape((1,-1))
  
  st_idx = np.where(valid_diff==1)[1] + 1
  end_idx = np.where(valid_diff==-1)[1]
    
  if st_idx.shape[0] > end_idx.shape[0]:
    end_idx = np.concatenate((end_idx,np.array([len(conditions)])))
  elif st_idx.shape[0] < end_idx.shape[0]:
    st_idx = np.concatenate((np.array([0]),st_idx))
  elif st_idx[0] > end_idx[0]:
    st_idx = np.concatenate((np.array([0]),st_idx))
    end_idx = np.concatenate((end_idx,np.array([len(conditions)])))
    
  section_list = []
  for i in range(st_idx.shape[0]):
    section_list.append((st_idx[i], end_idx[i]+1))
    
  return section_list

# Get calcium trace
def get_trace(data_dict, seg_id, scan_id, trace_type="spike"):
	seg_ids = data_dict["segment_id"]
	scan_ids = data_dict["scan_id"]
	traces = data_dict[trace_type]

	valid = (seg_ids==seg_id)*(scan_ids==scan_id)

	return traces[np.where(valid)[0][0]]

# get stimulus label
def get_stim_label(data_dict, scan_id):
	scan_ids = data_dict["scan_id"]
	conditions = data_dict["condition"]

	return np.round(conditions[scan_ids==scan_id][0],1)

def get_peakamp_tdarray(trace, condition):
  valid = ~np.isnan(condition)
  angle_list = np.unique(condition[valid])

  tdarray = np.zeros((30,16))
  for i in range(angle_list.shape[0]):

    angle = angle_list[i]
    section_list = get_section(condition, angle)

    offset = 0
    for j in range(len(section_list)):

      if len(section_list)!=30:
        offset = 30-len(section_list)
      
      s = section_list[j] 
      trace_sect = trace[s[0]:s[1]]

      max_idx = np.argmax(trace_sect)
      max_val = trace_sect[max_idx]
      if (max_idx==0):
        tdarray[j+offset,i] = 0      
      elif (max_idx==trace_sect.shape[0]-1):
        trace_post = trace[s[1]:s[1]+15]
        tdarray[j+offset,i] = np.max(trace_post)
      elif trace_sect[0]>0.5*max_val:
        tdarray[j+offset,i] = 0
      else:
        tdarray[j+offset,i] = max_val

  return tdarray

def tuning_curve(tdarray):
  u, s, vh = np.linalg.svd(tdarray, full_matrices=False)
  tune = np.abs(vh[0,:])
  
  return tune

def get_dsi(tune): 
  tune = tune.reshape((-1,1))
  angles = np.linspace(0, 2*np.pi, 17)[:-1]
  vec = np.zeros((1,16), dtype="complex")
  for i in range(16):
    vec[0,i] = np.exp(complex(0,angles[i]))
  K = vec@tune/np.sum(tune)
    
  return np.abs(K)[0]

def get_osi(tune):
  tune = tune.reshape((-1,1))
  angles = np.linspace(0, 4*np.pi, 17)[:-1]
  vec = np.zeros((1,16), dtype="complex")
  for i in range(16):
    vec[0,i] = np.exp(complex(0,angles[i]))
  K = vec@tune/np.sum(tune)
    
  return np.abs(K)[0]

# Permutation test
def shuffle_amparr(amp_arr): 
  amp_arr_copy = np.copy(amp_arr)
  for i in range(amp_arr.shape[0]):
    tmp = amp_arr_copy[i,:]
    np.random.shuffle(tmp)
      
  return amp_arr_copy

def permutation_test(amp_arr, n_shuff, mode="dsi"):
  si_shuffled = np.zeros(n_shuff)
    
  if mode=="dsi":
    tune_true = tuning_curve(amp_arr)    
    si_true = get_dsi(tune_true)
    for t in range(n_shuff):
      amp_arr_shuffled = shuffle_amparr(amp_arr)
      tune_shuffled = tuning_curve(amp_arr_shuffled)
      si_shuffled[t] = get_dsi(tune_shuffled)
  elif mode=="osi":
    tune_true = tuning_curve(amp_arr)
    si_true = get_osi(tune_true)
    for t in range(n_shuff):
      amp_arr_shuffled = shuffle_amparr(amp_arr)
      amp_arr_copy = np.copy(amp_arr)
      for i in range(amp_arr.shape[0]):   
        tmp = amp_arr_copy[i,:]
        np.random.shuffle(tmp)
        
      tune_shuffled = tuning_curve(amp_arr_shuffled)
      si_shuffled[t] = get_osi(tune_shuffled)
            
  p = percentileofscore(si_shuffled, si_true)
  p = 1-p/100
        
  return p

def osi_dsi_tuning(neuron_id):
  filtered_synapse_data = pd.read_csv('../data/filtered_synapse_data.csv')

  neuron_id_list = EASETuning["segment_id"]
  scan_list = EASETuning["scan_id"]
  scan_id = int(scan_list[neuron_id_list==neuron_id])
  trace = get_trace(EASETrace, neuron_id, scan_id, "trace_raw")
  stimlab = get_stim_label(Stimulus, scan_id)
  
  response_array = get_peakamp_tdarray(trace, stimlab)
  tune = tuning_curve(response_array)
  
  dsi = get_dsi(tune)
  osi = get_osi(tune)
  dsi_pvalue = permutation_test(response_array, 10000, "dsi")
  osi_pvalue = permutation_test(response_array, 10000, "osi")

  angles = []
  for value in stimlab[~np.isnan(stimlab)]:
      if value not in angles:
          angles.append(value)
  print(angles)

  orientation_ind = np.argmax(tune)
  best_orientation = angles[orientation_ind]  # Find direction with max response

  if osi > 0.5 and osi_pvalue < 0.01:
    osi_activity = 'High'
  else:
    osi_activity = 'Low'

  mask = filtered_synapse_data['post_root_id'] == neuron_id
  filtered_synapse_data.loc[mask, 'osi'] = osi
  filtered_synapse_data.loc[mask, 'osi_p'] = osi_pvalue
  filtered_synapse_data.loc[mask, 'dsi'] = dsi
  filtered_synapse_data.loc[mask, 'dsi_p'] = dsi_pvalue
  filtered_synapse_data.loc[mask, 'scan'] = scan_id
  filtered_synapse_data.loc[mask, 'osi_activity'] = osi_activity
  filtered_synapse_data.loc[mask, 'orientation_angle'] = best_orientation  
  
  matching_rows = filtered_synapse_data[filtered_synapse_data['post_root_id'] == neuron_id]
  print(matching_rows)