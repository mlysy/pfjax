# --- rough test code for proj_data --------------------------------------------
#
# Assumes the existence of a function `obj_fun(theta)`.

import pandas as pd
import seaborn as sns

# set theta limits and names
theta_lims = np.array([[3., 8.], [0., .1], [.5, 2]])
theta_names = ["mu", "sigma", "tau"]
n_pts = 100  # number of evaluation points per coordinate

# calculate projection plot
plot_df = proj_data(obj_fun,
                    theta, theta_lims, theta_names)

# plot using seaborn
sns.relplot(
    data=plot_df, kind="line",
    x="x", y="y", col="theta",
    facet_kws=dict(sharex=False, sharey=False)
)
