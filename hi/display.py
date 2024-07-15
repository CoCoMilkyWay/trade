import numpy as np
import matplotlib.pyplot as plt

# Define the dimensions of the image
width, height = 1024, 512

# Read the binary data from the files
with open('luma.bin', 'rb') as luma_file:
    luma_data = luma_file.read()

with open('chroma.bin', 'rb') as chroma_file:
    chroma_data = chroma_file.read()

# Convert the binary data to numpy arrays
luma = np.frombuffer(luma_data, dtype=np.uint16).reshape((height, width))
chroma = np.frombuffer(chroma_data, dtype=np.uint8).reshape((height, width, 2))

# Extract the Cb and Cr components
Cb = chroma[:, :, 0]
Cr = chroma[:, :, 1]

# Convert YCbCr to RGB
def ycbcr_to_rgb(y, cb, cr):
    y = y.astype(np.float32)
    cb = cb.astype(np.float32) - 128
    cr = cr.astype(np.float32) - 128

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)

# Convert the image from YCbCr to RGB
rgb_image = ycbcr_to_rgb(luma, Cb, Cr)
print(luma)
print(Cb)
print(Cr)

# Display the image
plt.imshow(rgb_image)
plt.axis('off')  # Hide the axis
plt.show()
