def to_one_hot(index, size):

    '''
    Int/list(Int), Int -> list(Int)/list(list(Int))

    Converts integral class representation to a one-hot class
    representation. <size> is the total number of classes.
    <index> can be passed as an integer or list of integers.
    '''
    try:
        index = int(index)
        output = [0 for _ in range(size)]
        output[index] = 1
        return output
    except:
        return to_one_hot_array(index, size)

def to_one_hot_array(index, size):

    '''
    list(Int), Int -> list(list(Int))

    Converts a list of integral class representation to a
    list of one-hot class representation.
    '''
    f = lambda x: to_one_hot(x, size)
    output = list()
    for i in index:
        output.append(f(i))
    return output

def ctrlflow_onehot(switch, index, size):

    '''
    Bool, Int/list(Int), Int -> Int/list(Int)/list(list(Int))
    
    A wrapper function to control whether the index is converted
    to one-hot representation or remain unchanged.
    Converts to one-hot if and only if switch=True.

    See to_one_hot for more details.
    '''
    if switch:
        return to_one_hot(index, size)
    else:
        return index

def from_one_hot(lst):
    '''
    list(Int)/list(list(Int)) -> Int/list(Int)

    Requires:
    * There is only one 1 in each data point

    Converts one-hot class representation to a integral class
    representation.
    '''
    if type(lst) is not list:
        lst = list(lst)
    try:
        lst = [list(i).index(1) for i in lst]
        return lst
    except:
        return lst.index(1)
    
