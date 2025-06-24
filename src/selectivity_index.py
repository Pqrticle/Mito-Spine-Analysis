import pickle

with open("../data/EASETuning.pkl", "rb") as f:
    EASETuning = pickle.load(f)

# The following neuron IDs are missing crucial mitochondria labels and must be excluded
exclude_ids = [648518346349538721, 648518346349536769, 648518346349539895, 648518346349538440]

def get_selectivity_indexes():
    segment_ids = EASETuning["segment_id"]
    osi_values = EASETuning["osi"]
    dsi_values = EASETuning["dsi"]
    selectivity_indexes = {
        int(seg_id): [float(osi), float(dsi)]
        for seg_id, osi, dsi in zip(segment_ids, osi_values, dsi_values)
        if int(seg_id) not in exclude_ids
    }
    
    return selectivity_indexes