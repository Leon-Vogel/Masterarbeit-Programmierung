import pickle

def fast_deepcopy(data_object): # way faster than copy.deepcopy!
    return pickle.loads(pickle.dumps(data_object))