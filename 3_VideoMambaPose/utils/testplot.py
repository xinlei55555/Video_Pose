import matplotlib.pyplot as plt

# List of points
points = [
    [80.0287, 84.8717],
    [70.1154, 203.5275],
    [75.2885, 41.0240],
    [41.4858, 102.4573],
    [101.3144, 89.2001],
    [50.9714, 236.1715],
    [82.5714, 226.8571],
    [81.5968, 169.3342],
    [156.2141, 124.5747],
    [46.5148, 322.8954],
    [108.4944, 305.7119],
    [150.8253, 163.8124],
    [200.4367, 102.0617],
    [5.1725, 389.7732],
    [93.2928, 367.7371]
]

# Extract x and y values
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color='blue')

# Set plot limits if needed
plt.xlim(0, 320)
plt.ylim(0, 240)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Points')

# Show the plot
plt.show()
