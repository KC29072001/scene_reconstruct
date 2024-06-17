import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import measure 
from scipy.ndimage import distance_transform_edt

# Preprocess the input 2D image 
def preprocess_image(image):
    # Convert image to grayscale 
    if len(image.shape) > 2:
        image = image.mean(axis=2)
    # Perform thresholding to obtain a binary image
    threshold = 0.5  
    binary_image = image > threshold
    return binary_image


# Compute the Signed Distance Field
def compute_signed_distance_field(binary_image):
    # Compute the distance transform of the binary image then SDF
    distance_transform = distance_transform_edt(binary_image)
    signed_distance_field = distance_transform - 0.5
    return signed_distance_field



# Reconstruct the 3D surface from the Signed Distance Field
def reconstruct_3d_surface(signed_distance_field):
    # Use marching cubes to extract the surface mesh from the signed distance field
    try:
        vertices, faces, _, _ = measure.marching_cubes(signed_distance_field, level=0)
    except ValueError as e:
        print("Error during marching cubes:", e)
        raise

    return vertices, faces



# Load the 2D image
image = plt.imread('demo2.png')

# Preprocess the 2D image
binary_image = preprocess_image(image)


# signed_distance_field is a 2D array with shape (3612, 6000)
signed_distance_field_2d = compute_signed_distance_field(preprocess_image(binary_image))

# Replicate the 2D array along the depth dimension
depth = 10  # Adjustable
signed_distance_field = np.repeat(np.expand_dims(signed_distance_field_2d, axis=2), depth, axis=2)

# Reconstruct the 3D surface
vertices, faces = reconstruct_3d_surface(signed_distance_field)


num_slices = 10
z_values = np.linspace(vertices[:, 2].min(), vertices[:, 2].max(), num_slices)
fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
for i, z in enumerate(z_values):
    slice_vertices = vertices[vertices[:, 2] == z, :]
    axes[i].scatter(slice_vertices[:, 0], slice_vertices[:, 1], c='b', s=1)
    axes[i].set_title(f'Z = {z:.2f}')
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    axes[i].set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()