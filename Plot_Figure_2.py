import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- Helper Functions ---

def df_to_grid(df):
    """
    Converts the DataFrame columns into 2D grids for pcolormesh and contourf.
    """
    # Extract unique sorted values for axes to form the grid
    tc2_vals = np.sort(df['tc2'].unique())
    tf2_vals = np.sort(df['tf2'].unique())
    
    # Create meshgrid for plotting coordinates
    T_C2, T_F2 = np.meshgrid(tc2_vals, tf2_vals)
    
    # Pivot data to form the Z-values matrix
    # We use pivot_table. The index (y-axis) is tf2, columns (x-axis) is tc2.
    # .values converts the pivoted dataframe to a numpy array.
    winding_grid = df.pivot_table(index='tf2', columns='tc2', values='winding number').values
    HL_grid = df.pivot_table(index='tf2', columns='tc2', values='HL').values
    HR_grid = df.pivot_table(index='tf2', columns='tc2', values='HR').values
    
    return T_C2, T_F2, winding_grid, HL_grid, HR_grid

def get_topo_label_text(tc2, tf2, t_c1, t_f1):
    """
    Calculates the topological label string (e.g., "1 \oplus 0") for a specific point.
    This reconstructs the logic for annotation purposes.
    """
    # Fixed parameters as per the original problem
    v1, v2 = -0.3, 0.3
    
    t_cf1 = (t_c1 - t_f1) / 2.0
    t_cf2_prime = (tc2 + tf2) / 2.0
    v12 = (v1 - v2) / 2.0 
    
    if np.isclose(t_cf2_prime, 0): return "?"

    R = np.sqrt(t_cf2_prime**2 + v12**2)
    s = np.sign(t_cf2_prime)
    
    # Recalculate u and v for L and R sectors
    # Note: t_cf2 is (tc2 - tf2)/2
    t_cf2_diff = (tc2 - tf2) / 2.0
    
    u_L = t_cf1 - s * R
    v_L = t_cf2_diff + s * R
    u_R = t_cf1 + s * R 
    v_R = t_cf2_diff - s * R

    is_L = 1 if np.abs(u_L) < np.abs(v_L) else 0
    is_R = 1 if np.abs(u_R) < np.abs(v_R) else 0
    
    return f"${is_L} \oplus {is_R}$"

def plot_single_panel(ax, df, t_c1, t_f1, title_label, show_ylabel=True, show_legend=True):
    """
    Plots a single phase diagram panel on the provided axes.
    """
    # 1. Prepare Data
    T_C2, T_F2, winding_grid, HL_grid, HR_grid = df_to_grid(df)
    
    # 2. Plot Winding Number Heatmap
    im = ax.pcolormesh(T_C2, T_F2, winding_grid, cmap='coolwarm', shading='gouraud', vmin=-2, vmax=0)
    
    # 3. Overlay Hatched Regions
    # HL Topological Region
    ax.contourf(T_C2, T_F2, HL_grid, levels=[0.5, 1.5], colors='none', hatches=['///'], extend='neither')
    # HR Topological Region
    ax.contourf(T_C2, T_F2, HR_grid, levels=[0.5, 1.5], colors='none', hatches=['\\\\\\'], extend='neither')
    
    # 4. Plot Symmetry Line: tf2 = -(tc1 + tf1) - tc2
    line_x = np.array([T_C2.min(), T_C2.max()]) 
    line_y = -(t_c1 + t_f1) - line_x
    
    # Plot line only within view limits
    ax.plot(line_x, line_y, color='red', linestyle='--', linewidth=3, label=r'$t_{c_1}+t_{f_1}+t_{c_2}+t_{f_2}=0$')
    
    # 5. Add Annotations along the line
    tc2_points = [-2.0, -0.2, 2.0]
    for tc2_pt in tc2_points:
        tf2_pt = -(t_c1 + t_f1) - tc2_pt
        
        # Only annotate if point is within the y-axis range roughly
        if T_F2.min() < tf2_pt < T_F2.max():
            label_text = get_topo_label_text(tc2_pt, tf2_pt, t_c1, t_f1)
            ax.text(tc2_pt, tf2_pt, label_text, 
                    fontsize=14, ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # 6. Formatting
    ax.set_xlim(T_C2.min(), T_C2.max())
    ax.set_ylim(T_F2.min(), T_F2.max())
    
    ax.set_xlabel(r'$t_{c_2}$', fontsize=18)
    if show_ylabel:
        ax.set_ylabel(r'$t_{f_2}$', fontsize=18)
        
    ax.text(-0.1, 1.05, title_label, transform=ax.transAxes, fontsize=18, fontweight='bold')
    
    # 7. Legend
    if show_legend:
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', hatch='///', label=r'$H_L$ Topological'),
            Patch(facecolor='white', edgecolor='black', hatch='\\\\\\', label=r'$H_R$ Topological'),
            Line2D([0], [0], color='red', ls='--', lw=2, label=r'$t_{c_1}+t_{f_1}+t_{c_2}+t_{f_2}=0$')
        ]
        ax.legend(handles=legend_elements, facecolor='white', frameon=True, loc='upper right', fontsize=10)
        
    return im

# --- Main Script ---

# Load Data
try:
    df_a = pd.read_excel('a.xlsx')
    df_b = pd.read_excel('b.xlsx')
except FileNotFoundError:
    print("Error: Make sure 'a.xlsx' and 'b.xlsx' are in the current directory.")
    exit()

# Setup Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot Panel (a): t_c1=1.0, t_f1=0.9
im1 = plot_single_panel(axes[0], df_a, t_c1=1.0, t_f1=0.9, title_label='(a)', show_legend=True)

# Plot Panel (b): t_c1=1.0, t_f1=-0.9
im2 = plot_single_panel(axes[1], df_b, t_c1=1.0, t_f1=-0.9, title_label='(b)', show_ylabel=False, show_legend=False)

# Add Colorbar
fig.subplots_adjust(right=0.85, wspace=0.1)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
cb = fig.colorbar(im1, cax=cbar_ax, ticks=[-2, -1, 0])
cb.set_label(r'$\nu$', fontsize=18)
cb.ax.tick_params(labelsize=14)

# Save and Show
output_filename = 'reproduced_phase_diagram.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
print(f"Figure saved as {output_filename}")
plt.show()
