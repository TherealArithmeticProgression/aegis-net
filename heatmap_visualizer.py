import matplotlib.pyplot as plt
import numpy as np

# Dummy data for landing spots and safety scores
landing_spots = np.array([[1, 2], [2, 6], [3, 1], [4, 5], [5, 2]])  # Example landing spots (x, y)
safety_scores = np.array([0.5, 0.8, 0.2, 0.9, 0.7])  # Safety scores for each landing spot

# Create a heatmap
heatmap, xedges, yedges = np.histogram2d(landing_spots[:,0], landing_spots[:,1], bins=10, weights=safety_scores)

# Plotting the heatmap
plt.clf()  # Clear the current figure
plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
plt.colorbar(label='Safety Score')
plt.title('Landing Spots Safety Score Heatmap')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.show()