import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left


# --- Linear interpolation ---
def linear_interp(x_list, y_list, x):
    return np.interp(x, x_list, y_list)

# --- Hagan DF: linear in discount factors (smooth direct) ---
def hagan_df_interp(x_list, y_list, x):
    x_sorted = np.argsort(x_list)
    x_list, y_list = x_list[x_sorted], y_list[x_sorted]
    i = np.searchsorted(x_list, x) - 1
    i = np.clip(i, 0, len(x_list) - 2)
    t0, t1 = x_list[i], x_list[i+1]
    p0, p1 = y_list[i], y_list[i+1]
    return p0 + (p1 - p0) * (x - t0) / (t1 - t0)

# --- Hagan: linear in forward rate (integrate smooth forward) ---
def hagan_interp(x_list, y_list, x):
    """
    Hagan exact interpolation:
    - Build interval forwards g_i
    - Build node forwards F_k (weighted average)
    - On [t_i, t_{i+1}] integrate linear f(t) exactly to get P(x)
    Assumes x_list, y_list are floats and y_list are discount factors > 0.
    Extrapolation: constant nearest DF.
    """

    # Convert to numpy arrays and sort by x
    x_arr = np.array(x_list, dtype=float)
    y_arr = np.array(y_list, dtype=float)
    if len(x_arr) < 2:
        raise ValueError("Need at least two points for Hagan interpolation.")
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    # quick bounds handling (constant extrapolation)
    if x <= x_arr[0]:
        return float(y_arr[0])
    if x >= x_arr[-1]:
        return float(y_arr[-1])

    # interval widths
    deltas = np.diff(x_arr)  # length n-1

    # interval forward rates g_i for each interval [t_i, t_{i+1}]
    g = -np.log(y_arr[1:] / y_arr[:-1]) / deltas  # length n-1

    # node forwards F_k (length n)
    n = len(x_arr)
    F = np.empty(n, dtype=float)
    F[0] = g[0]
    F[-1] = g[-1]
    if n > 2:
        # weighted average: F_k = (g_{k-1}*dt_{k-1} + g_k*dt_k) / (dt_{k-1}+dt_k)
        for k in range(1, n - 1):
            w1 = deltas[k - 1]
            w2 = deltas[k]
            F[k] = (g[k - 1] * w1 + g[k] * w2) / (w1 + w2)
    else:
        # only two nodes -> F are both equal to g[0]
        F[1] = F[0]

    # find interval i such that x in [x_i, x_{i+1}]
    i = bisect_left(x_arr, x) - 1
    i = max(0, min(i, n - 2))
    t0 = x_arr[i]
    t1 = x_arr[i + 1]
    P0 = y_arr[i]
    dt = t1 - t0
    tau = x - t0

    Fi = F[i]
    Fi1 = F[i + 1]

    # exact integral of linear forward between t0 and x:
    # integral = Fi * tau + 0.5 * (Fi1 - Fi) * tau^2 / dt
    exponent = - (Fi * tau + 0.5 * (Fi1 - Fi) * (tau ** 2) / dt)
    Px = P0 * np.exp(exponent)
    return float(Px)



# Example discount curve data
x_list = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
y_list = np.array([1.0, 0.98, 0.97, 0.90, 0.66])  # Discount factors

# --- Build fine grid ---
x_vals = np.linspace(0, 5, 200)
y_linear = [linear_interp(x_list, y_list, x) for x in x_vals]
y_hagan_df = [hagan_df_interp(x_list, y_list, x) for x in x_vals]
y_hagan = [hagan_interp(x_list, y_list, x) for x in x_vals]

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(x_list, y_list, 'o', label='Data points', color='black')
plt.plot(x_vals, y_linear, '--', label='Linear', color='red')
plt.plot(x_vals, y_hagan_df, '-', label='Hagan DF (approx)', color='green')
plt.plot(x_vals, y_hagan, '-', label='Hagan (smooth fwd)', color='blue')
plt.xlabel("Time (years)")
plt.ylabel("Discount Factor")
plt.title("Linear vs Hagan DF vs Hagan Interpolation")
plt.legend()
plt.grid(True)
plt.show()
