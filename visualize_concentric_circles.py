"""
Concentric Circles Visualization for QA Difficulty Levels
Each difficulty level shown as expanding circles from target table
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
import numpy as np

# Set up the figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('QA Difficulty Levels: Concentric Circle Model',
             fontsize=24, fontweight='bold', y=0.98)

# Color scheme
COLORS = {
    'target': '#2ECC71',      # Green - target table
    'hop1': '#3498DB',        # Blue - 1-hop tables
    'hop2': '#E74C3C',        # Red - 2-hop tables
    'hop3': '#9B59B6',        # Purple - 3-hop tables
    'bg_easy': '#D5F4E6',     # Light green background
    'bg_medium': '#AED6F1',   # Light blue background
    'bg_hard': '#F5B7B1',     # Light red background
}

def draw_concentric_pattern(ax, num_hops, pattern_type='path', difficulty='E', title=''):
    """
    Draw concentric circles showing query complexity

    Args:
        num_hops: 0, 1, 2, 3 (number of JOIN hops)
        pattern_type: 'path' (chain) or 'star' (intersection)
        difficulty: 'E', 'M', 'H'
    """
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Background color based on difficulty
    if num_hops == 0:
        bg_color = COLORS['bg_easy']
    elif num_hops == 1:
        bg_color = COLORS['bg_medium']
    else:
        bg_color = COLORS['bg_hard']

    ax.set_facecolor(bg_color)

    # Draw concentric circles (from outer to inner for proper layering)
    radii = [4.0, 3.0, 2.0, 1.0, 0.5]

    # Draw outer circles (decorative)
    for i in range(num_hops, -1, -1):
        if i < len(radii):
            circle = Circle((0, 0), radii[i],
                          fill=True,
                          facecolor='white',
                          edgecolor='gray',
                          linewidth=2,
                          alpha=0.8,
                          zorder=1)
            ax.add_patch(circle)

    # Draw target table (center)
    target_circle = Circle((0, 0), 0.6,
                          fill=True,
                          facecolor=COLORS['target'],
                          edgecolor='black',
                          linewidth=3,
                          zorder=10)
    ax.add_patch(target_circle)
    ax.text(0, 0, 'Orders\n(Target)',
           ha='center', va='center',
           fontsize=11, fontweight='bold',
           zorder=11)

    # Draw nodes based on pattern
    if num_hops == 0:
        # Direct query - no external tables
        pass

    elif num_hops == 1:
        # Single hop
        if pattern_type == 'path':
            # 1p: chain
            node_x, node_y = 2.5, 0
            circle = Circle((node_x, node_y), 0.5,
                          facecolor=COLORS['hop1'],
                          edgecolor='black',
                          linewidth=2.5,
                          zorder=8)
            ax.add_patch(circle)
            ax.text(node_x, node_y, 'Customers',
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   zorder=9)

            # Arrow
            arrow = FancyArrowPatch((0.6, 0), (node_x - 0.5, node_y),
                                  arrowstyle='<->', mutation_scale=25,
                                  linewidth=3, color='black',
                                  zorder=7)
            ax.add_patch(arrow)

    elif num_hops == 2:
        if pattern_type == 'path':
            # 2p: chain
            # First hop
            node1_x, node1_y = 2.0, 0
            circle1 = Circle((node1_x, node1_y), 0.45,
                           facecolor=COLORS['hop1'],
                           edgecolor='black',
                           linewidth=2.5,
                           zorder=8)
            ax.add_patch(circle1)
            ax.text(node1_x, node1_y, 'Customers',
                   ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   zorder=9)

            # Second hop
            node2_x, node2_y = 3.5, 0
            circle2 = Circle((node2_x, node2_y), 0.45,
                           facecolor=COLORS['hop2'],
                           edgecolor='black',
                           linewidth=2.5,
                           zorder=8)
            ax.add_patch(circle2)
            ax.text(node2_x, node2_y, 'Regions',
                   ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   zorder=9)

            # Arrows
            arrow1 = FancyArrowPatch((0.6, 0), (node1_x - 0.45, node1_y),
                                   arrowstyle='<->', mutation_scale=20,
                                   linewidth=2.5, color='black',
                                   zorder=7)
            ax.add_patch(arrow1)

            arrow2 = FancyArrowPatch((node1_x + 0.45, node1_y),
                                   (node2_x - 0.45, node2_y),
                                   arrowstyle='<->', mutation_scale=20,
                                   linewidth=2.5, color='black',
                                   zorder=7)
            ax.add_patch(arrow2)

        else:  # star/intersection
            # 2i: star with 2 tables
            angles = [np.pi/4, -np.pi/4]  # Top-right and bottom-right
            labels = ['Customers', 'Products']
            colors = [COLORS['hop1'], COLORS['hop1']]

            for angle, label, color in zip(angles, labels, colors):
                node_x = 2.5 * np.cos(angle)
                node_y = 2.5 * np.sin(angle)

                circle = Circle((node_x, node_y), 0.5,
                              facecolor=color,
                              edgecolor='black',
                              linewidth=2.5,
                              zorder=8)
                ax.add_patch(circle)
                ax.text(node_x, node_y, label,
                       ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       zorder=9)

                # Arrow from center to node
                dx = node_x - 0
                dy = node_y - 0
                dist = np.sqrt(dx**2 + dy**2)
                start_x = 0.6 * dx / dist
                start_y = 0.6 * dy / dist
                end_x = node_x - 0.5 * dx / dist
                end_y = node_y - 0.5 * dy / dist

                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                      arrowstyle='<->', mutation_scale=25,
                                      linewidth=3, color='black',
                                      zorder=7)
                ax.add_patch(arrow)

    elif num_hops == 3:
        if pattern_type == 'star':
            # 3i: star with 3 tables
            angles = [np.pi/2, np.pi/6, -np.pi/6]  # Top, top-right, bottom-right
            labels = ['Customers', 'Products', 'Sales Reps']
            colors = [COLORS['hop1'], COLORS['hop1'], COLORS['hop1']]

            for angle, label, color in zip(angles, labels, colors):
                node_x = 2.8 * np.cos(angle)
                node_y = 2.8 * np.sin(angle)

                circle = Circle((node_x, node_y), 0.48,
                              facecolor=color,
                              edgecolor='black',
                              linewidth=2.5,
                              zorder=8)
                ax.add_patch(circle)

                # Adjust text size for longer labels
                fontsize = 8.5 if len(label) > 10 else 9.5
                ax.text(node_x, node_y, label,
                       ha='center', va='center',
                       fontsize=fontsize, fontweight='bold',
                       zorder=9)

                # Arrow from center to node
                dx = node_x - 0
                dy = node_y - 0
                dist = np.sqrt(dx**2 + dy**2)
                start_x = 0.6 * dx / dist
                start_y = 0.6 * dy / dist
                end_x = node_x - 0.48 * dx / dist
                end_y = node_y - 0.48 * dy / dist

                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                      arrowstyle='<->', mutation_scale=22,
                                      linewidth=3, color='black',
                                      zorder=7)
                ax.add_patch(arrow)

    # Add title and difficulty label
    ax.text(0, 4.7, title,
           ha='center', va='top',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor='black',
                    linewidth=2))

    # Add difficulty indicator
    diff_colors = {'E': '#A8E6CF', 'M': '#FFD93D', 'H': '#FF6B6B'}
    diff_labels = {'E': 'Easy', 'M': 'Medium', 'H': 'Hard'}

    if difficulty in diff_colors:
        ax.text(0, -4.7, f'{difficulty} - {diff_labels[difficulty]}',
               ha='center', va='bottom',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor=diff_colors[difficulty],
                        edgecolor='black',
                        linewidth=2))


# ============================================================================
# Draw each difficulty level
# ============================================================================

# Row 1: Simple patterns
# 0E: Direct
draw_concentric_pattern(axes[0, 0], num_hops=0, pattern_type='path',
                       difficulty='E', title='0E - Direct Query')
axes[0, 0].text(0, -3.5, 'No JOIN required\nFilter on target table only',
               ha='center', va='center',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 1pE: Single-hop Path
draw_concentric_pattern(axes[0, 1], num_hops=1, pattern_type='path',
                       difficulty='E', title='1pE - Single-hop Path')
axes[0, 1].text(0, -3.5, '1 JOIN (chain)\nSimple filter conditions',
               ha='center', va='center',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 1pM: Single-hop Path Medium
draw_concentric_pattern(axes[0, 2], num_hops=1, pattern_type='path',
                       difficulty='M', title='1pM - Single-hop Path')
axes[0, 2].text(0, -3.5, '1 JOIN (chain)\nMultiple filter conditions',
               ha='center', va='center',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Row 2: Complex patterns
# 2pE: Two-hop Path
draw_concentric_pattern(axes[1, 0], num_hops=2, pattern_type='path',
                       difficulty='E', title='2pE - Two-hop Chain')
axes[1, 0].text(0, -3.5, '2 JOINs (chain)\nTransitive reasoning',
               ha='center', va='center',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 2iE: Star-2 Intersection
draw_concentric_pattern(axes[1, 1], num_hops=2, pattern_type='star',
                       difficulty='E', title='2iE - Two-way Intersection')
axes[1, 1].text(0, -3.5, '2 JOINs (star)\nMulti-dimensional filter',
               ha='center', va='center',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 3iE: Star-3 Intersection
draw_concentric_pattern(axes[1, 2], num_hops=3, pattern_type='star',
                       difficulty='E', title='3iE - Three-way Intersection')
axes[1, 2].text(0, -3.5, '3 JOINs (star)\nComplex multi-dimensional',
               ha='center', va='center',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['target'], edgecolor='black',
                  label='Target Table (Orders)', linewidth=2),
    mpatches.Patch(facecolor=COLORS['hop1'], edgecolor='black',
                  label='1-hop Foreign Tables', linewidth=2),
    mpatches.Patch(facecolor=COLORS['hop2'], edgecolor='black',
                  label='2-hop Foreign Tables', linewidth=2),
]

fig.legend(handles=legend_elements, loc='lower center',
          fontsize=12, ncol=3, frameon=True,
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('/home/user/Talk2Metadata/concentric_circles_difficulty.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Concentric circles visualization saved to: concentric_circles_difficulty.png")
plt.show()
