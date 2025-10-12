def is_empty(v):
    # This is important; the def of what is empty. Be careful about this in the future!!!
    if v is None:
        return True
    if isinstance(v, str) and v == "": 
        return True
    if isinstance(v, (list, tuple, dict, set)) and len(v) == 0:
        return True
    return False

def prune_dict(d):
    return {
        k: v
        for k, v in d.items()
        # drop if key is empty-string or value is “empty”:
        if (k != "") and (not is_empty(v))
    }   


