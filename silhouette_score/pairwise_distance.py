from affinegap import normalizedAffineGapDistance

def pairwise_affine_gap_distance(x1,x2):
    """Calculate normalized affine gap distance between two samples x1 and x2.
    Parameters
    ----------
    x1: sample 1, array [n_features]
    x2: sample 2, array [n_features]
    Returns
    -------
    agap_distance: float, normalized affine gap distance of a given pair
    """
    agap_distance = 0
    assert(len(x1)==len(x2))
    for i in range(len(x1)):
        str1 = str(x1[i]) if x1[i] else '' 
        str2 = str(x2[i]) if x2[i] else '' 
        if not str1 and not str2:
            agap_distance += 0.5
        else:
            agap_distance += normalizedAffineGapDistance(str1, str2)
    return agap_distance
