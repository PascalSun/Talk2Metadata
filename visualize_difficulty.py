"""
Visualization of QA Difficulty Classification System
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Set up the figure with subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Color scheme
COLORS = {
    'anchor': '#6B9BD1',      # Blue - anchor nodes
    'variable': '#F4A261',    # Orange - variable nodes
    'answer': '#90C695',      # Green - answer nodes
    'easy': '#A8E6CF',        # Light green
    'medium': '#FFD93D',      # Yellow
    'hard': '#FF6B6B',        # Red
    'expert': '#845EC2',      # Purple
}

def draw_node(ax, x, y, label, node_type='answer', size=0.3):
    """Draw a single node"""
    color = COLORS[node_type]
    circle = Circle((x, y), size, color=color, ec='black', linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=10,
            fontweight='bold', zorder=4)

def draw_arrow(ax, x1, y1, x2, y2):
    """Draw an arrow between nodes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black', zorder=2)
    ax.add_patch(arrow)

# ============================================================================
# PANEL 1: Pattern Types (Schema Diagrams)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 18)
ax1.set_ylim(0, 4)
ax1.axis('off')
ax1.set_title('Pattern Types (Query Graph Structures)', fontsize=16, fontweight='bold', pad=20)

# 0: Direct
x_offset = 1
draw_node(ax1, x_offset, 2, 'Orders', 'answer', 0.4)
draw_node(ax1, x_offset, 0.8, 'status=\n"done"', 'anchor', 0.35)
draw_arrow(ax1, x_offset, 1.3, x_offset, 1.6)
ax1.text(x_offset, 3.2, '0 (Direct)', ha='center', fontsize=12, fontweight='bold')
ax1.text(x_offset, 0.1, 'No JOIN', ha='center', fontsize=9, style='italic')

# 1p: Single-hop Path
x_offset = 4
draw_node(ax1, x_offset, 2, 'Orders', 'answer', 0.4)
draw_node(ax1, x_offset + 1.2, 2, 'Customers', 'variable', 0.4)
draw_node(ax1, x_offset + 2.4, 2, 'industry=\n"Health"', 'anchor', 0.35)
draw_arrow(ax1, x_offset + 0.5, 2, x_offset + 0.65, 2)
draw_arrow(ax1, x_offset + 1.75, 2, x_offset + 1.9, 2)
ax1.text(x_offset + 1.2, 3.2, '1p (Single-hop Path)', ha='center', fontsize=12, fontweight='bold')
ax1.text(x_offset + 1.2, 0.1, '1 JOIN (chain)', ha='center', fontsize=9, style='italic')

# 2p: Two-hop Path
x_offset = 8.5
draw_node(ax1, x_offset, 2, 'Orders', 'answer', 0.35)
draw_node(ax1, x_offset + 1, 2, 'Customers', 'variable', 0.35)
draw_node(ax1, x_offset + 2, 2, 'Regions', 'variable', 0.35)
draw_node(ax1, x_offset + 3, 2, 'name=\n"US-W"', 'anchor', 0.35)
draw_arrow(ax1, x_offset + 0.4, 2, x_offset + 0.6, 2)
draw_arrow(ax1, x_offset + 1.4, 2, x_offset + 1.6, 2)
draw_arrow(ax1, x_offset + 2.4, 2, x_offset + 2.6, 2)
ax1.text(x_offset + 1.5, 3.2, '2p (Two-hop Path)', ha='center', fontsize=12, fontweight='bold')
ax1.text(x_offset + 1.5, 0.1, '2 JOINs (chain)', ha='center', fontsize=9, style='italic')

# 2i: Star-2 (Intersection)
x_offset = 14.5
draw_node(ax1, x_offset + 1, 2, 'Orders', 'answer', 0.4)
draw_node(ax1, x_offset + 1, 3.2, 'Customers', 'variable', 0.35)
draw_node(ax1, x_offset + 1, 0.8, 'Products', 'variable', 0.35)
draw_node(ax1, x_offset + 2.2, 3.2, 'industry=\n"Health"', 'anchor', 0.35)
draw_node(ax1, x_offset + 2.2, 0.8, 'category=\n"SW"', 'anchor', 0.35)
draw_arrow(ax1, x_offset + 1, 2.5, x_offset + 1, 2.8)
draw_arrow(ax1, x_offset + 1, 1.5, x_offset + 1, 1.2)
draw_arrow(ax1, x_offset + 1.4, 3.2, x_offset + 1.8, 3.2)
draw_arrow(ax1, x_offset + 1.4, 0.8, x_offset + 1.8, 0.8)
ax1.text(x_offset + 1, 3.7, '2i (Two-way Intersection)', ha='center', fontsize=12, fontweight='bold')
ax1.text(x_offset + 1, 0.1, '2 JOINs (star)', ha='center', fontsize=9, style='italic')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['answer'], edgecolor='black', label='Answer (Target Table)'),
    mpatches.Patch(facecolor=COLORS['variable'], edgecolor='black', label='Variable (Foreign Table)'),
    mpatches.Patch(facecolor=COLORS['anchor'], edgecolor='black', label='Anchor (Filter Value)'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)

# ============================================================================
# PANEL 2: Difficulty Spectrum (Score Distribution)
# ============================================================================
ax2 = fig.add_subplot(gs[1, :])

# Define all difficulty codes
difficulties = {
    # Easy tier
    "0E": 0.0, "0M": 0.3, "0H": 0.6,
    # Medium tier
    "1pE": 1.0, "1pM": 1.3, "1pH": 1.6,
    # Hard tier
    "2pE": 2.0, "2pM": 2.3, "2pH": 2.6,
    "2iE": 2.0, "2iM": 2.3, "2iH": 2.6,
    # Expert tier
    "3pE": 3.0, "3pM": 3.3, "3pH": 3.6,
    "3iE": 3.0, "3iM": 3.3, "3iH": 3.6,
    "4iE": 4.0, "4iM": 4.3, "4iH": 4.6,
}

# Sort by score
sorted_diffs = sorted(difficulties.items(), key=lambda x: x[1])
codes, scores = zip(*sorted_diffs)

# Color mapping
def get_color(score):
    if score < 1.0:
        return COLORS['easy']
    elif score < 2.0:
        return COLORS['medium']
    elif score < 3.0:
        return COLORS['hard']
    else:
        return COLORS['expert']

colors = [get_color(s) for s in scores]

# Create bar chart
bars = ax2.barh(range(len(codes)), scores, color=colors, edgecolor='black', linewidth=1.5)

# Add labels on bars
for i, (code, score) in enumerate(zip(codes, scores)):
    ax2.text(score + 0.1, i, f'{code} ({score:.1f})',
            va='center', fontsize=9, fontweight='bold')

ax2.set_yticks(range(len(codes)))
ax2.set_yticklabels(codes, fontsize=9)
ax2.set_xlabel('Difficulty Score', fontsize=12, fontweight='bold')
ax2.set_title('Difficulty Spectrum (All Codes)', fontsize=16, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(0, 5.5)

# Add tier separators and labels
tier_positions = [1.0, 2.0, 3.0]
tier_labels = ['Easy', 'Medium', 'Hard', 'Expert']
tier_y = len(codes)

for pos in tier_positions:
    ax2.axvline(x=pos, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Add tier background colors
ax2.axvspan(0, 1.0, alpha=0.1, color=COLORS['easy'])
ax2.axvspan(1.0, 2.0, alpha=0.1, color=COLORS['medium'])
ax2.axvspan(2.0, 3.0, alpha=0.1, color=COLORS['hard'])
ax2.axvspan(3.0, 5.5, alpha=0.1, color=COLORS['expert'])

# Add tier labels at top
for i, (start, end, label) in enumerate([(0, 1, 'Easy'), (1, 2, 'Medium'),
                                          (2, 3, 'Hard'), (3, 5.5, 'Expert')]):
    mid = (start + end) / 2
    ax2.text(mid, -1.5, label, ha='center', fontsize=11,
            fontweight='bold', style='italic')

# ============================================================================
# PANEL 3: Examples Table
# ============================================================================
ax3 = fig.add_subplot(gs[2, :])
ax3.axis('off')
ax3.set_title('Concrete Examples', fontsize=16, fontweight='bold', pad=20)

# Create example table
examples = [
    ["0E", "Direct, Easy", "Find orders with status='completed'"],
    ["0M", "Direct, Medium", "Find orders with amount>10K AND status='completed'"],
    ["1pE", "1-hop Path, Easy", "Find orders from Healthcare customers"],
    ["1pM", "1-hop Path, Medium", "Find orders from Healthcare customers with revenue>1M"],
    ["2pE", "2-hop Chain, Easy", "Find orders from US-West region customers"],
    ["2iE", "Star-2, Easy", "Find orders: Healthcare customers × Software products"],
    ["2iM", "Star-2, Medium", "Find orders: US Healthcare customers × Enterprise Software"],
    ["3iE", "Star-3, Easy", "Find orders: Healthcare customers × Software products × Senior sales rep"],
]

# Table parameters
table_top = 0.9
row_height = 0.11
col_widths = [0.08, 0.22, 0.70]
col_positions = [0.05, 0.13, 0.35]

# Header
header_color = '#E8E8E8'
for i, (pos, width, text) in enumerate(zip(col_positions, col_widths,
                                            ['Code', 'Description', 'Example Question'])):
    box = FancyBboxPatch((pos, table_top), width, row_height,
                         boxstyle="round,pad=0.01",
                         facecolor=header_color, edgecolor='black', linewidth=2)
    ax3.add_patch(box)
    ax3.text(pos + width/2, table_top + row_height/2, text,
            ha='center', va='center', fontsize=11, fontweight='bold')

# Data rows
for row_idx, (code, desc, example) in enumerate(examples):
    y_pos = table_top - (row_idx + 1) * row_height

    # Determine row color based on code
    score = difficulties.get(code, 0)
    row_color = get_color(score)

    # Draw cells
    for col_idx, (pos, width, text) in enumerate(zip(col_positions, col_widths,
                                                      [code, desc, example])):
        box = FancyBboxPatch((pos, y_pos), width, row_height,
                            boxstyle="round,pad=0.005",
                            facecolor=row_color, edgecolor='gray', linewidth=1, alpha=0.6)
        ax3.add_patch(box)

        # Adjust text properties
        if col_idx == 0:  # Code column
            ax3.text(pos + width/2, y_pos + row_height/2, text,
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    family='monospace')
        else:
            ax3.text(pos + 0.01, y_pos + row_height/2, text,
                    ha='left', va='center', fontsize=9)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# ============================================================================
# Overall Title
# ============================================================================
fig.suptitle('Talk2Metadata QA Difficulty Classification System',
             fontsize=20, fontweight='bold', y=0.98)

# Add subtitle with format
fig.text(0.5, 0.95, 'Format: {Pattern}{Difficulty}  |  Pattern: 0, 1p, 2p, 2i, 3p, 3i, 4i  |  Difficulty: E (Easy), M (Medium), H (Hard)',
         ha='center', fontsize=11, style='italic', color='#555555')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/user/Talk2Metadata/difficulty_classification_system.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Visualization saved to: difficulty_classification_system.png")
plt.show()
