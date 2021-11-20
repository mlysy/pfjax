import numpy as np
import pandas as pd


def proj_data(fun, theta, theta_lims, theta_names, n_pts=100):
    """
    Create a DataFrame of projection plot data.

    For each element `i` of `theta`, evaluates `lambda x: fun([theta[:i], x, theta[i:]])` 
    for `x` linearly spaced between `theta_lims[i,0]` and `theta_lims[i,1]`.

    Args:
        fun: Function with argument `theta` returning a scalar.
        theta: Values onto which to project `fun()`.
        theta_lims: ndarray of size `theta.size x 2` giving the projection plot limits.
        theta_names: Names of the elements of `theta`.
        n_pts: Number of linearly spaced evaluation points.

    Returns:
        A DataFrame with columns:
            - id: Integers between 0 and `n_pts-1` giving the index of the point on the plot.
            - theta: Parameter name indicating which projection plot the data is for.
            - x: x-coordinate of the projection plot.
            - y: y-coordinate of the projection plot.
    """
    n_theta = theta_lims.shape[0]
    plot_data = np.empty((2, n_theta, n_pts))
    for i in range(n_theta):
        x_theta = np.linspace(theta_lims[i][0], theta_lims[i][1], num=n_pts)
        for j in range(n_pts):
            theta_tmp = np.copy(theta)
            theta_tmp[i] = x_theta[j]
            plot_data[0, i, j] = x_theta[j]
            plot_data[1, i, j] = fun(theta_tmp)
    col_names = ["x", "y"]
    return pd.merge(*[(pd.DataFrame(plot_data[i].T,
                                    columns=theta_names)
                       .assign(id=range(n_pts))
                       .melt(id_vars="id", var_name="theta", value_name=col_names[i])
                       ) for i in range(2)], on=["id", "theta"])
