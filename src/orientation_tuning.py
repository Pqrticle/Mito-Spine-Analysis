import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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

def permutation_test(amp_arr, n_shuff):
  si_shuffled = np.zeros(n_shuff)
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

def get_stimlab_scan_id(neuron_id):
  neuron_id_list = EASETuning["segment_id"]
  scan_list = EASETuning["scan_id"]

  scan_indices = neuron_id_list == neuron_id
  if np.any(scan_indices):  # Checks if there's at least one True value in the condition
    scan_id = int(scan_list[scan_indices][0])
  else:
    return None, None

  stimlab = get_stim_label(Stimulus, scan_id)
  return stimlab, scan_id

def osi_tuning(neuron_id):
  stimlab, scan_id = get_stimlab_scan_id(neuron_id)
  if scan_id is None:
    return None, None, None

  trace = get_trace(EASETrace, neuron_id, scan_id, "trace_raw")
  response_array = get_peakamp_tdarray(trace, stimlab)
  tune = tuning_curve(response_array)

  osi = get_osi(tune)
  osi_pvalue = permutation_test(response_array, 10000)

  angles = []
  for value in stimlab[~np.isnan(stimlab)]:
    if value not in angles:
      angles.append(value)
  orientation_ind = np.argmax(tune)
  best_orientation = angles[orientation_ind]  # Find direction with max response

  return osi, osi_pvalue, best_orientation

def process_osi_data():
  filtered_synapse_data = pd.read_csv('../data/filtered_synapse_data.csv')

  unique_ids = list(set(filtered_synapse_data["pre_root_id"]).union(set(filtered_synapse_data["post_root_id"])))
  extra_data_list = []
  osi_values = {}

  for neuron_id in unique_ids:
    osi, osi_pvalue, orientation = osi_tuning(neuron_id)
    if osi is not None:
      osi_values[neuron_id] = osi
      extra_data_list.append({'neuron_id': neuron_id, 'osi': osi, 'osi_pvalue': osi_pvalue, 'osi_orientation': orientation})

  # Convert osi_values to numpy array
  osi_values_array = np.array(list(osi_values.values())).reshape(-1, 1)

  # Perform KMeans clustering
  kmeans = KMeans(n_clusters=2, random_state=42)  # Two clusters: low and high activity
  osi_labels = kmeans.fit_predict(osi_values_array)

  # Create a new dictionary with neuron_id as key and activity (High/Low) as value
  activity_dict = {
    neuron_id: 'High' if osi_labels[idx] == 1 else 'Low'
    for idx, neuron_id in enumerate(osi_values.keys())
  }

  # Add the activity information to extra_data_list
  for data in extra_data_list:
    neuron_id = data['neuron_id']
    data['osi_activity'] = activity_dict[neuron_id]

  extra_data_df = pd.DataFrame(extra_data_list)
  extra_data_df.to_csv('../data/osi_data.csv', index=False)

  filtered_synapse_data['pre_osi_activity'] = filtered_synapse_data['pre_root_id'].map(
      lambda x: activity_dict.get(x, 'No OSI Data')
  )
  filtered_synapse_data['post_osi_activity'] = filtered_synapse_data['post_root_id'].map(
      lambda x: activity_dict.get(x, 'No OSI Data')
  )

  filtered_synapse_data.to_csv('../data/filtered_synapse_data.csv', index=False)

process_osi_data()