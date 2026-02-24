from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerBase
import numpy as np
from matplotlib.colors import to_rgba


# Colorblind friendly colors from https://sronpersonalpages.nl/~pault/
CB_COLORS = ['#EE6677', '#4477AA', '#CCBB44', '#228833', '#66CCEE', '#AA3377', ]
# Create a cmap with them
CMAP = ListedColormap(CB_COLORS)


class CombinedLegendObject:
    """ A custom legend object that combines multiple handles into one. """
    def __init__(self, handles):
        self.handles = [Line2D([0], [0], marker='o', color=handle.get_facecolor()[0], markersize=5)
                        for handle in handles]


class HandlerCombinedLegendObject(HandlerBase):
    """ A custom legend handler for CombinedLegendObject. """
    def create_artists(self, legend, orig_handle, x_descent, y_descent, width, height, fontsize, trans):
        handles = orig_handle.handles
        for i, handle in enumerate(handles):
            handle.set_data([width * (i + 1) / (len(handles) + 1)], [height / 2])
            handle.set_transform(trans)
        return handles


def rotation_matrix(theta_deg):
    """ Returns the 2D rotation matrix for a given angle in degrees. """
    theta = np.deg2rad(theta_deg)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


def line_points(P, n, ax0, ax1, t_range=(-100, 100)):
    """ Returns two points on the line defined by point P and normal n in the plane defined by axes ax0 and ax1. """
    t_values = np.linspace(*t_range, 2)
    x = P[ax0] + t_values * -n[ax1]
    y = P[ax1] + t_values * n[ax0]
    return x, y


def linear_infeasible(xx, yy, P, n, ax0, ax1):
    """ Returns a boolean mask where the linear constraint defined by point P and normal n is infeasible (i.e., the
    half-space opposite to the normal). """
    return (xx - P[ax0]) * n[ax0] + (yy - P[ax1]) * n[ax1] > 0


def quadratic_infeasible(xx, yy, H2, c2, b):
    """ Returns a boolean mask where the quadratic constraint defined by H2, c2, b is infeasible (i.e., outside the
    ellipse). """
    pts = np.stack([xx.ravel(), yy.ravel()])
    diff = pts.T - c2
    vals = np.einsum('ij,jk,ik->i', diff, H2, diff).reshape(xx.shape)
    return vals > b


def plot_linear(ax, constraint, ax0, ax1, color, label=None, shade=False, grid=None, alpha=0.1, line_style='--'):
    """ Plots a linear constraint defined by point P and normal n. Optionally shades the infeasible region. """
    P, n = constraint['P'], constraint['n']
    x, y = line_points(P, n, ax0, ax1)
    ax.plot(x, y, linestyle=line_style, color=color, label=label)

    if shade and grid is not None:
        xx, yy = grid
        mask = linear_infeasible(xx, yy, P, n, ax0, ax1)
        rgba = to_rgba(color, alpha=alpha)
        ax.contourf(xx, yy, mask, levels=[0.5, 1], colors=[rgba], alpha=alpha)


def plot_ellipse(ax, H2, c2, b, **kwargs):
    """ Plots an ellipse on the matplotlib axis ax. """
    eigvals, eigvecs = np.linalg.eigh(H2)
    axis_lengths = np.sqrt(b / eigvals)

    theta = np.linspace(0, 2 * np.pi, 100000)
    circle = np.array([np.cos(theta), np.sin(theta)])

    ellipse_scaled = np.diag(axis_lengths) @ circle
    ellipse_rotated = eigvecs @ ellipse_scaled
    ellipse_points = ellipse_rotated + c2[:, np.newaxis]

    h = ax.plot(ellipse_points[0, :], ellipse_points[1, :], **kwargs)
    return h


def plot_quadratic(ax, constraint, ax0, ax1, color, label=None, shade=False, grid=None, alpha=0.1, line_style='--'):
    """ Plots a quadratic constraint defined by H, c, b. Optionally shades the infeasible region. """
    H, c, b = constraint['H'], constraint['c'], constraint['b']
    H2 = H[np.ix_([ax0, ax1], [ax0, ax1])]
    c2 = c[[ax0, ax1]]

    plot_ellipse(ax, H2, c2, b, linestyle=line_style, color=color, label=label)

    if shade and grid is not None:
        xx, yy = grid
        mask = quadratic_infeasible(xx, yy, H2, c2, b)
        rgba = to_rgba(color, alpha=alpha)
        ax.contourf(xx, yy, mask, levels=[0.5, 1], colors=[rgba], alpha=alpha)


def plot_linear_constraints(ax, constraints, ax0, ax1, cmap, shade, grid):
    """ Plots multiple linear constraints. """
    multiple = len(constraints) > 1
    for k, con in enumerate(constraints):
        label = f'Linear con. {k + 1}' if multiple else 'Linear con.'
        color = cmap(k)
        plot_linear(ax, con, ax0, ax1, color, label=label, shade=shade, grid=grid)


def plot_quadratic_constraints(ax, constraints, ax0, ax1, cmap, shade, grid, base_index=0):
    """ Plots multiple quadratic constraints. """
    multi = len(constraints) > 1
    for k, con in enumerate(constraints):
        label = f'Convex-quadratic con. {k + 1}' if multi else 'Convex-quadratic con.'
        color = cmap(base_index + k)
        plot_quadratic(ax, con, ax0, ax1, color, label=label, shade=shade, grid=grid)


def shade_multi_constraint(ax, multi_constraint, ax0, ax1, cmap, color_index, grid, alpha=0.1):
    """ 
    Shades infeasible region for a single multi-constraint:
    infeasible = AND_over_groups( OR_over_constraints_in_group(mask) ).
    """
    xx, yy = grid
    infeasible_all = np.full_like(xx, True, dtype=bool)

    for group in multi_constraint:
        group_mask = np.full_like(xx, False, dtype=bool)
        for con in group.get('Linear', []):
            group_mask |= linear_infeasible(xx, yy, con['P'], con['n'], ax0, ax1)
        for con in group.get('Quadratic', []):
            H2 = con['H'][np.ix_([ax0, ax1], [ax0, ax1])]
            c2 = con['c'][[ax0, ax1]]
            b = con['b']
            group_mask |= quadratic_infeasible(xx, yy, H2, c2, b)
        infeasible_all &= group_mask

    rgba = to_rgba(cmap(color_index), alpha=alpha)
    ax.contourf(xx, yy, infeasible_all, levels=[0.5, 1], colors=[rgba], alpha=alpha)


def plot_multi_constraints(ax, multi_constraints, ax0, ax1, cmap, shade, grid, single_label=True, start_index=0):
    """ Plots multiple multi-constraints. """
    color_index = start_index
    for idx, multi in enumerate(multi_constraints):
        base_name = 'Multi con.' + (f' {idx + 1}' if idx > 0 else '')
        if shade:
            shade_multi_constraint(ax, multi, ax0, ax1, cmap, color_index, grid)

        # Draw each group’s primitives (labeling rules preserved)
        for g, group in enumerate(multi):
            if single_label:
                group_label = base_name
                first_label = (g == 0)
            else:
                group_label = base_name + (f' Group {g + 1} ' if len(multi) > 1 else '')
                first_label = True

            # Use the same color for the whole multi unless single_label=False
            group_color_index = color_index if single_label else color_index + g
            color = cmap(group_color_index)

            # Linear in group
            for k, con in enumerate(group.get('Linear', [])):
                label = group_label if first_label else None
                # shaded already by union mask
                plot_linear(ax, con, ax0, ax1, color, label=label, shade=False, grid=None)
                first_label = False

            # Quadratic in group
            for k, con in enumerate(group.get('Quadratic', [])):
                label = group_label if first_label else None
                plot_quadratic(ax, con, ax0, ax1, color, label=label, shade=False, grid=None)
                first_label = False

        color_index += 1
