import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.rand(100, 2)

grid_rows = 10
grid_cols = 10
input_dim = 2
initial_learning_rate = 0.2
num_epochs = 200

initial_radius = max(grid_rows, grid_cols) / 2
time_constant = num_epochs / np.log(initial_radius)

weight_matrix = np.random.rand(grid_rows, grid_cols, input_dim)

x, y = np.meshgrid(np.arange(grid_rows), np.arange(grid_cols))
neuron_locations = np.stack([x, y], axis=2)

for epoch in range(num_epochs):

    learning_rate = initial_learning_rate * np.exp(-epoch / num_epochs)
    radius = initial_radius * np.exp(-epoch / time_constant)

    for input_vector in data:

        distances = np.linalg.norm(weight_matrix - input_vector, axis=2)
        bmu_index = np.argmin(distances)
        bmu_coords = np.unravel_index(bmu_index, (grid_rows, grid_cols))

        bmu_location = np.array(bmu_coords)
        neuron_distance = np.linalg.norm(neuron_locations - bmu_location, axis=2)

        influence = np.exp(-(neuron_distance**2) / (2 * (radius**2)))

        influence = influence[:, :, np.newaxis]
        weight_matrix += influence * learning_rate * (input_vector - weight_matrix)


plt.figure(figsize=(8, 8))


plt.imshow(np.mean(weight_matrix, axis=2), cmap='viridis')
plt.colorbar(label="Weight Intensity")


plt.scatter(
    data[:, 0] * (grid_rows - 1),
    data[:, 1] * (grid_cols - 1),
    color='red',
    label='Data Points'
)

plt.legend()
plt.title("Self-Organizing Map (SOM)")
plt.show()
