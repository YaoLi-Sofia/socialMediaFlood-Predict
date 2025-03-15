import numpy as np

#  acha ice model parameters
beta_degree_ice = 3

# --- water microphysical model terms
beta_degree_water = 3

qe_006um_coef_water = np.array([2.39378, -0.39669, 0.10725], 'f4')

re_beta_110um_coef_water = np.array([0.59356, 1.41647, -1.12240, 0.53016], 'f4')
qe_110um_coef_water = np.array([-1.08658, 3.95746, -1.18558], 'f4')
wo_110um_coef_water = np.array([-0.11394, 0.77357, -0.23835], 'f4')
g_110um_coef_water = np.array([0.23866, 0.97914, -0.31347], 'f4')

beta_110um_133um_coef_water = np.array([1.03152, 1.31866, -0.13426, -0.02558], 'f4')
