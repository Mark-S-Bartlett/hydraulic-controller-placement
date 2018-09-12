import numpy as np

def differentiated_linear_weights(dist, bins=40):
    mask = (dist != 0)
    hist, bin_edges = np.histogram(dist[mask].ravel(), range=(0,dist.max()+1e-5), bins=bins)
    dig = np.digitize(dist, bin_edges[1:])
    weights = (hist / hist.max())[dig]
    weights[~mask] = 0
    return weights

def differentiated_power_weights(dist, power=2, bins=40):
    weights = differentiated_linear_weights(dist, bins=bins)
    weights = weights**power
    return weights

def threshold_weights(dist, threshold=0.85, keep_weights=True):
    weights = differentiated_linear_weights(dist)
    if keep_weights:
        weights = np.where(weights >= threshold, weights, 0).astype(int).astype(float)
    else:
        weights = (weights >= threshold).astype(int).astype(float)
    return weights

def controller_placement_algorithm(fdir, c, k, weights, grid, compute_weights, dist_weights,
                                       weight_args={}, dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
    '''
    c : max accumulation allowed at cut
    k : number of partitions
    '''
    mfdir = fdir.copy()
    subs = []
    dirs = []
    ixes = []
    kwargs = {'affine' : grid.affine,
              'shape' : grid.shape,
              'crs' : grid.crs,
              'nodata_in' : 0}

    acc = grid.accumulation(data=mfdir, dirmap=dirmap, inplace=False, **kwargs)
    wacc = grid.accumulation(data=mfdir, weights=weights, dirmap=dirmap, inplace=False,
                             **kwargs)
    pour_point_y, pour_point_x = np.unravel_index(np.argmax(acc), grid.shape)
    nonsub = grid.view('catch')

    for i in range(k):
        if hasattr(c, "__len__"):
            ci = c[i]
        else:
            ci = c
        # Recompute distance histogram
        sub_dist_weights = np.where(nonsub.ravel(), dist_weights, 0)
        dist = grid.flow_distance(data=nonsub, x=pour_point_x, y=pour_point_y,
                                  weights=sub_dist_weights,
                                  inplace=False, dirmap=dirmap, **kwargs)
        dist[np.isnan(dist)] = 0
        weights = compute_weights(dist, **weight_args)
        acc = grid.accumulation(data=mfdir, dirmap=dirmap, inplace=False, **kwargs)
        wacc = grid.accumulation(data=mfdir, weights=weights, dirmap=dirmap, inplace=False,
                                 **kwargs)
        if acc.max() == 0:
            break
        ix = np.argmax(np.where(acc < ci, wacc, 0))
        # Specify pour point
        y, x = np.unravel_index(ix, wacc.shape)
        # Delineate the catchment
        sub = grid.catchment(x, y, data=mfdir, dirmap=dirmap, recursionlimit=15000,
                             xytype='index', inplace=False, **kwargs)
        ixes.append(ix)
        subs.append(sub)
        mfdir = np.where((sub != 0), 0, mfdir).astype(np.uint8)
        nonsub = np.where((sub != 0), 0, grid.view('catch'))
    return subs, ixes

def naive_partition(fdir, target_cells, k, grid, size_range=[0,1], tolerance_spread=100, max_errors=10,
                    use_seed=True, seed_0=0, seed_1=0, dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
    '''
    c : max accumulation allowed at cut
    k : number of partitions
    '''
    mfdir = fdir.copy()
    subs = []
    dirs = []
    ixes = []
    kwargs = {'affine' : grid.affine,
              'shape' : grid.shape,
              'crs' : grid.crs,
              'nodata_in' : 0}

    acc = grid.accumulation(data=mfdir, dirmap=dirmap, inplace=False, **kwargs)
    nonsub = grid.view('catch')
    # Hack
    ix = 0
    numcells = 0
    err_counter = 0

    for i in range(k):
        # Recompute distance histogram
        acc = grid.accumulation(data=mfdir, dirmap=dirmap, inplace=False, **kwargs)
        if acc.max() <= 1:
            break
        if numcells >= target_cells:
            break

        if i == (k - 1):
            close_to_acc = np.abs(acc - (target_cells - numcells)).ravel()
            ix = np.argmin(close_to_acc)
        else:
            if use_seed:
                np.random.seed(seed_0)
            c = np.random.randint(*size_range)
            close_to_acc = np.abs(acc - c).ravel()
            close_to_acc_ix = np.where(close_to_acc < tolerance_spread)[0]
            try:
                if use_seed:
                    np.random.seed(seed_1)
                ix = close_to_acc_ix[np.random.randint(close_to_acc_ix.size)]
            except:
                i -= 1
                err_counter += 1
                print('Failed to find subcatchment')
                if err_counter >= max_errors:
                    break
                else:
                    continue
        numcells += acc.flat[ix]
        # Specify pour point
        y, x = np.unravel_index(ix, acc.shape)
        # Delineate the catchment
        sub = grid.catchment(x, y, data=mfdir, dirmap=dirmap, recursionlimit=15000,
                             xytype='index', inplace=False, **kwargs)
        ixes.append(ix)
        subs.append(sub)
        mfdir = np.where((sub != 0), 0, mfdir).astype(np.uint8)
        nonsub = np.where((sub != 0), 0, grid.view('catch'))
    return subs, ixes
