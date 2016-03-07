def line_search_bisection(f, bound, accuracy):
    """
    Maximize c so that constraint fulfilled.
    
    This algorithm assumes continuity of f; that is,
    there exists a fixed value c, such that f(x) is 
    False for x < c and True otherwise. This holds true,
    for example, for the levelsets that we consider.
    
    Parameters
    ----------
    f - callable
        A function that takes a scalar value and return True if
        the constraint is fulfilled, False otherwise.
    bound - list
        Interval within which to search
    accuracy - float
        The interval up to which the algorithm shall search
        
    Returns
    -------
    c - float
        The maximum value c so that the constraint is fulfilled    
    """
    # Break if lower bound does not fulfill constraint
    if not f(bound[0]):
        return None
    
    if f(bound[1]):
        return bound[1]
    
    while bound[1] - bound[0] > accuracy:
        mean = (bound[0] + bound[1]) / 2
        
        if f(mean):
            bound[0] = mean
        else:
            bound[1] = mean
    
    return bound[0]
