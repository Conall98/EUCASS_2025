# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 10:47:30 2025

@author: cdepaor
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example synthetic data (replace with your launch years)
DB = pd.read_excel(r"C:\Users\cdepaor2\Desktop\EUCASS_2025 - Archive\lander year list.xlsx")
data = DB["year"]  # years of lunar landers

# Define bins
bins = np.arange(1960, 2026, 5)  # (1960,1965], ..., (2020,2025]

# Plot histogram
fig, ax = plt.subplots(figsize=(10,6))
counts, edges, patches = ax.hist(
    data, bins=bins, edgecolor='black', color='#006658', zorder=2
)

# Title
ax.set_title("Landers Launched Towards The Moon", fontsize=18)

# Set xticks at *all* bin edges (including rightmost edge)
ax.set_xticks(edges)

# Add bin edge labels rotated at 45 degrees
ax.set_xticklabels([str(int(edge)) for edge in edges], rotation=45, fontsize=11)

# Y-axis: only multiples of 5
max_y = int(max(counts))
ax.set_yticks(np.arange(0, max_y+5, 5))
ax.tick_params(axis='y', labelsize=11)  # y-axis font size

# Gridlines (behind bars)
ax.grid(axis="y", linestyle="--", color="grey", alpha=0.7, zorder=1)

# Add counts on top of bars (skip zero counts)
for count, patch in zip(counts, patches):
    if count > 0:
        ax.text(
            patch.get_x() + patch.get_width()/2,
            count + 0.2,
            str(int(count)),
            ha='center', va='bottom', fontsize=14
        )

# Axis labels font size
ax.tick_params(axis='x', labelsize=11)

plt.tight_layout()
plt.show()
