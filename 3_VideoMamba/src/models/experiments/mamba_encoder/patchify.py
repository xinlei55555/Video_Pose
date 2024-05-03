import numpy as np

def patchify_image(image, patch_size):
    """
    Patchify the input image into smaller patches.

    Args:
    - image: numpy array representing the input image.
    - patch_size: integer representing the size of each patch.

    Returns:
    - patches: numpy array containing the flattened patches.
    - num_patches: integer representing the number of patches.
    """
    # Get image dimensions
    height, width, channels = image.shape

    # Calculate number of patches in each dimension
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Initialize array to store patches
    patches = np.zeros((num_patches_h * num_patches_w, patch_size * patch_size * channels))

    # Extract patches and flatten
    patch_index = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :].flatten()
            patches[patch_index, :] = patch
            patch_index += 1

    return patches, num_patches_h * num_patches_w

def reshape_patches(patches, num_patches, patch_size):
    """
    Reshape the patches back into a grid.

    Args:
    - patches: numpy array containing the flattened patches.
    - num_patches: integer representing the number of patches.
    - patch_size: integer representing the size of each patch.

    Returns:
    - reshaped_image: numpy array representing the reshaped image grid.
    """
    # Calculate grid dimensions
    grid_size = int(np.sqrt(num_patches))
    grid_height = grid_width = grid_size * patch_size

    # Initialize array to store reshaped image
    reshaped_image = np.zeros((grid_height, grid_width, patches.shape[1] // (patch_size * patch_size)))

    # Reshape patches back into grid
    patch_index = 0
    for i in range(grid_size):
        for j in range(grid_size):
            patch = patches[patch_index, :].reshape((patch_size, patch_size, -1))
            reshaped_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = patch
            patch_index += 1

    return reshaped_image

# Example usage
image = np.random.rand(224, 224, 3)  # Example input image
patch_size = 4  # Example patch size

# Patchify the image
patches, num_patches = patchify_image(image, patch_size)

# Reshape patches back into a grid
reshaped_image = reshape_patches(patches, num_patches, patch_size)

# Check if the reshaped image matches the original image
print(np.allclose(image, reshaped_image))
print(patches.shape)
print(reshaped_image.shape)
