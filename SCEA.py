import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import haversine_distances

# ===============================================================================


def SCEA(
    point_coordinates,
    point_values,
    growth_limit=2,
    detection_limit=3.5,
    radius_func="default",
    n_clusters="auto",
    point_value_threshold="stds_from_median",
    distance_matrix="euclidean",
    max_points_in_start_radius=7,
    local_box_size=0,
    verbose=True,
):
    """
    Finds n clusters.

    It starts the clustering with the maximum point.
    It considers only points in local_box_size * local_box_size around the maximum point. This makes distance calculations much faster.
    Also that local areas point_values gets standardised to mean value 0 and standard deviation to 1.
        Every point that gets added to a cluster starts as a "radiaiting point". It means that the point can add new points to the cluster.
    For each iteration, every radiaiting point starts its radius equal to the distance to the closest point not in the cluster.
    If there are more than max_points_in_start_radius in the starting radius, that point stops radiating.
    Otherwise the starting radius gets multiplied by radius_func output which is a function of point_value[i]. Larger point_value -> larger radius.
    Every point that falls in the radius gets added to the cluster. If no points fall in to the radius then that point stops radiating.
    This continues until all points stop radiating and no new points get added to the cluster.
        After that it removes all the clustered points from the dataset and finds the new maximum point and starts clustering there.
    It keeps finding new clusters until a new maximum point is under the threshold. The threshold is stds of standard deviations away from the median/mean.

    Parameters:
    ----------
    points : array of shape (n,2)
        Coordinates of points.
    point_value : array of shape (n,)
        Values for every point. This gets fed in to the radius function.
    radius_func : lambda function or 'default'
        default: f(x) = np.min([1 + x - growth_limit , 2]).
        The radius func values that give values less than 1 dont capture new points,
         since the starting radius is equal to the distance of closest point not in the cluster
         and that radius gets multiplied by this functions output
    n_clusters : int, 'auto'
        int: Number of clusters to find. Detection_limit -parameter is ignored.
        "auto": automatically stop when the threshold has been met (determined by detection_limit)
    point_value_threshold : 'stds_from_median', 'stds_from_mean', float
        float: threshold value for stopping condition of new clusters. Detection_limit -parameter is ignored.
        stds_from_median: median value gets moved to 0 and standard deviation to 1, and after that finds the value "detection_limit" away from zero
        stds_from_mean: same but mean gets moved to 0
    detection_limit : float
        How many standard deviations away the stopping condition of new clusters threshold is.
    distance_matrix : 'euclidean', 'haversine', array of shape(n,n)
        euclidean: The natural Euclidean distance
        haversine: Distance on the surface of a ball. Here it its on the ball thats the size of the earth. So distance on earth approximately.
    growth_limit : float
        Parameter in the default radius_func.
        Since the point_values get standardised before using the function this value can be interperted as being
         the threshold value x standard deviations away where values bigger than this threshold capture neighbouring points.
    max_points_in_start_radius : int
        How many already clustered points can be in the starting radius of a 'radiating' point before it stops radiating.
    local_box_size : float
        How large of an area around the maximum points gets considered. If 0, all points are considered.
    verbose : boolean
        If True, it prints info.

    Returns:
    -------
    Integer array of shape (n,:).
     0: no cluster
     other integer: cluster number i
    """

    # Basic checks
    if (not isinstance(n_clusters, int)) and (not n_clusters == "auto"):
        raise Exception("n_clusters not valid")
    if np.isnan(point_values).sum() > 0:
        raise Exception("point_value has nan values")
    if len(point_coordinates) < 2:
        raise Exception("Size must be larger than 1")
    if len(point_coordinates) != len(point_values):
        raise Exception("points and point_value length must be the same")

    # Initialize arrays
    not_clustered = np.ones(len(point_values), dtype=bool)
    not_clustered_indexes = np.arange(len(point_values))
    clusters = np.zeros(
        len(point_values)
    )  # 0 = no cluster, otherwise the index of the cluster

    # Set threshold for automatic stopping condition
    if n_clusters == "auto":
        if point_value_threshold == "stds_from_mean":
            point_value_standardised = (
                preprocessing.StandardScaler()
                .fit_transform(point_values.reshape(-1, 1))
                .flatten()
            )
            point_value_standardised[point_value_standardised < detection_limit] = np.inf
            if np.isinf(point_value_standardised).sum() == len(
                point_value_standardised
            ):
                if verbose:
                    print(
                        "No clusters found. Consider lowering detection_limit=%f."
                        % detection_limit
                    )
                return clusters
            point_value_threshold = point_values[np.argmin(point_value_standardised)]
            if verbose:
                print("threshold value:", point_value_threshold)

        elif point_value_threshold == "stds_from_median":
            median, std = np.median(point_values), np.std(point_values)
            point_value_standardised = (point_values - median) / std
            point_value_standardised[point_value_standardised < detection_limit] = np.inf
            if np.isinf(point_value_standardised).sum() == len(
                point_value_standardised
            ):
                if verbose:
                    print(
                        "No clusters found. Consider lowering detection_limit=%f."
                        % detection_limit
                    )
                return clusters
            point_value_threshold = point_values[np.argmin(point_value_standardised)]
    else:
        point_value_threshold = -np.inf

    i = 0
    while True:
        points_not_clustered = point_coordinates[not_clustered]
        point_value_not_clustered = point_values[not_clustered]

        # Check stopping condition
        if isinstance(n_clusters, int) and i == n_clusters:
            break
        elif np.max(point_value_not_clustered) < point_value_threshold:
            if verbose:
                print("Found ", i, " clusters")
            break

        # Crop the area to a smaller box
        if local_box_size != 0:
            argmax = np.argmax(point_value_not_clustered)
            center_lon = points_not_clustered[argmax, 0]
            center_lat = points_not_clustered[argmax, 1]
            points_in_range = np.logical_and(
                np.logical_and(
                    points_not_clustered[:, 0] > center_lon - (local_box_size / 2),
                    points_not_clustered[:, 0] < center_lon + (local_box_size / 2),
                ),
                np.logical_and(
                    points_not_clustered[:, 1] > center_lat - (local_box_size / 2),
                    points_not_clustered[:, 1] < center_lat + (local_box_size / 2),
                ),
            )
            local_points = points_not_clustered[points_in_range]
            local_point_value = point_value_not_clustered[points_in_range]
            local_points_index = np.where(points_in_range)[0]
        # Or just use all points
        else:
            local_points = points_not_clustered
            local_point_value = point_value_not_clustered
            local_points_index = np.array(range(len(local_points)))

        # Cluster
        clt = find_one_cluster_v31(
            local_points,
            local_point_value,
            radius_func,
            growth_limit,
            distance_matrix,
            max_points_in_start_radius,
        )
        clusters[not_clustered_indexes[local_points_index[clt]]] = i + 1

        # Update not_clustered
        not_clustered = clusters == 0
        not_clustered_indexes = np.where(not_clustered)[0]

        i = i + 1

    return clusters


# ===============================================================================


def find_one_cluster_v31(
    points,
    point_value,
    radius_func="default",
    growth_limit=2,
    distance_matrix="euclidean",
    max_points_in_start_radius=7,
    preprocessor="Standard",
):
    """
    Stars with radius thats equal to distance to closest point, and then applies the radius function.

    """

    if radius_func == "default":
        radius_func = lambda x: np.min([1 + x - growth_limit, 2])

    # Preprocessing
    if preprocessor == "Standard":
        point_value = (
            preprocessing.StandardScaler()
            .fit_transform(point_value.reshape(-1, 1))
            .flatten()
        )
    elif preprocessor == "Quantile":
        point_value = (
            preprocessing.QuantileTransformer(
                output_distribution="uniform", n_quantiles=3
            )
            .fit_transform(point_value.reshape(-1, 1))
            .flatten()
        )
    elif preprocessor == "None":
        pass
    else:
        point_value = preprocessor.fit_transform(point_value.reshape(-1, 1)).flatten()

    # Distance matrix for points
    if isinstance(distance_matrix, str):
        if distance_matrix == "euclidean":
            distance_matrix = squareform(pdist(points))
        elif distance_matrix == "haversine":
            distance_matrix = (
                haversine_distances(np.radians(points)) * 6371000 / 1000
            )  # multiply by Earth radius to get kilometers
        else:
            raise Exception("Unknown distance_matrix option")

    # Initialize cluster
    cluster = np.zeros(len(points), dtype=bool)
    radiating_points = np.zeros(len(points), dtype=bool)

    # Find the index with max point_value value. Add it to the cluster
    argmax = np.argmax(point_value)
    cluster[argmax] = True
    radiating_points[argmax] = True

    # Find points around of added_points that are not already in the cluster. These are the new points.
    while True:
        not_old_points = np.logical_not(cluster)
        new_points = np.zeros(len(points), dtype=bool)  # initialize array with Falses

        # If there are no points left to add, break
        if not_old_points.sum() == 0:
            break

        # Find points added in this iteration
        for i in np.nonzero(radiating_points)[0]:
            # try:
            #    radius_of_closest_point_not_in_cluster = np.min(
            #        distance_matrix[i][not_old_points]
            #    )
            # except:
            #    pass
            radius_of_closest_point_not_in_cluster = np.min(
                distance_matrix[i][not_old_points]
            )
            min_index = np.min(
                [len(distance_matrix[i]) - 1, max_points_in_start_radius]
            )
            radius_of_nth_closest_point = np.partition(distance_matrix[i], min_index)[
                min_index
            ]

            points_in_radius = distance_matrix[
                i
            ] < radius_of_closest_point_not_in_cluster * radius_func(point_value[i])
            if (
                radius_of_closest_point_not_in_cluster < radius_of_nth_closest_point
            ) and (points_in_radius.sum() != 0):
                new_points = np.logical_or(
                    new_points, np.logical_and(points_in_radius, not_old_points)
                )
            else:
                # Stop radiating
                radiating_points[i] = False

        cluster = np.logical_or(new_points, cluster)  # Add new point to cluster
        radiating_points = np.logical_or(
            radiating_points, new_points
        )  # get index of added points

        # Stop when no new points are added
        if new_points.sum() == 0:
            break

    return cluster


# ===============================================================================


def filter_clusters(
    clusters,
    rg_points_dense,
    rg_points_value_matrix,
    rg_points_all,
    rg_points_value_flatten,
    regrid_square_size,
    min_cluster_size=15,
    max_neighbouring_nan=True,
    min_nan_neighbouring_ratio=0.5,
    return_dropped=False,
):
    """
    Cluster filtering

    This has three filters.
    1. Remove cluster if its too small, parametered by min_cluster_size.
    2. Remove cluster if the maximum value of cluster is next to nan values, parametered by max_neighbouring_nan.
    3. Remove cluster if too large ratio of clusters points are next to nan values, parametered by min_nan_neighbouring_ratio.
    This has been optimized to work with regridded data.

    Parameters:
    ----------
    clusters : array of shape (a*b,)
        all positive nonzero integers means a cluster
    rg_points_dense : grid points. List size of 2, where index 0 has unique x values (a,) and index 1 has unique y values (b,)
        sasd
    rg_points_value_matrix : grid points values in a matrix of shape (a,b)
        aasdf
    rg_points_all : array of shape (a*b,)
        all individual combinations of grids unique x and y values
    rg_points_value_flatten : array of shape (a*b,)
        values of points flatten
    regrid_square_size : float
        grid points distance to neighbouring points
    min_cluster_size : int
        clusters of size smaller than this get omitted
    max_neighbouring_nan : bool
        if true, omits clusters that have their max value next to nans
    min_nan_neighbouring_ratio : float in range [0,1]
        if cluster has larger ratio of point that neighbour nan values, then cluster gets omitted
    return_dropped : True
        If true returns also list of clusters numbers that got omitted

    Returns:
    ----------
    clusters_filtered : (a*b,)
        array with omitted clusters set to 0
    dropped : array
        numbers of clusters omitted
    """
    clusters_filtered = clusters.copy()
    r = regrid_square_size * 1.1
    not_nan_indx = np.where(~np.isnan(rg_points_value_flatten))[0]
    dropped = []
    for i in np.unique(clusters):
        if i == 0:
            continue

        # Filter if cluster is too small
        if min_cluster_size:
            if (clusters == i).sum() < min_cluster_size:
                clusters_filtered[clusters_filtered == i] = 0
                dropped.append(i)
                continue

        # Filter if max value of cluster is next to nans
        if max_neighbouring_nan:
            argmax = rg_points_value_flatten[not_nan_indx[clusters == i]].argmax()
            max_point = rg_points_all[not_nan_indx[clusters == i][argmax]]

            x_boolean = np.logical_and(
                rg_points_dense[0] < max_point[0] + r,
                rg_points_dense[0] > max_point[0] - r,
            )
            y_boolean = np.logical_and(
                rg_points_dense[1] < max_point[1] + r,
                rg_points_dense[1] > max_point[1] - r,
            )
            local_area = rg_points_value_matrix[np.ix_(x_boolean, y_boolean)]
            if np.isnan(local_area).sum() > 0:
                clusters_filtered[clusters_filtered == i] = 0
                dropped.append(i)
                continue

        # Filter if there is too much nan neighbours to the cluster
        if min_nan_neighbouring_ratio:
            points_with_nan_neighbours = 0
            cluster = rg_points_all[not_nan_indx[clusters == i]]
            cluster_size = len(cluster)
            for j in cluster:
                # local box
                x_boolean = np.logical_and(
                    rg_points_dense[0] < j[0] + r, rg_points_dense[0] > j[0] - r
                )
                y_boolean = np.logical_and(
                    rg_points_dense[1] < j[1] + r, rg_points_dense[1] > j[1] - r
                )
                local_area = rg_points_value_matrix[np.ix_(x_boolean, y_boolean)]
                if np.isnan(local_area).sum() > 0:
                    points_with_nan_neighbours = points_with_nan_neighbours + 1
            if points_with_nan_neighbours / cluster_size > min_nan_neighbouring_ratio:
                clusters_filtered[clusters_filtered == i] = 0
                dropped.append(i)

    if return_dropped:
        return clusters_filtered, dropped
    return clusters_filtered
