# import cv2
# import numpy as np

# # Load the image in grayscale to check the non-transparent parts
# image_path = '다운로드.png'
# image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# # Check if the image was successfully loaded
# if image is None:
#     raise ValueError("The image path is incorrect or the image is not accessible.")

# # Assuming the image has an alpha channel (transparency)
# # If the alpha channel is present, it will be the 4th channel in the image array
# if image.shape[2] == 4:
#     # Use the alpha channel as a mask
#     alpha_channel = image[:, :, 3]
    
#     # The areas where the mask is not zero, we want to keep, hence we convert it to 0 (black)
#     # The areas where the mask is zero, we want to set as 1 (white)
#     mask = np.where(alpha_channel > 0, 1, 0).astype(np.uint8)
#     # Save the mask to a file
#     mask_path = 'masked_output.png'
#     cv2.imwrite(mask_path, mask * 255)  # We multiply by 255 to convert from 0,1 to 0,255

#     mask_path
# else:
#     print("The image does not have an alpha channel or is not in the correct format.")

import cv2
import numpy as np

# Load the image in grayscale to check the non-transparent parts
image_path = r'C:\Users\User\Desktop\capstone\blended-latent-diffusion\inputs\23.png'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if the image was successfully loaded
if image is None:
    raise ValueError("The image path is incorrect or the image is not accessible.")

# Assuming the image has an alpha channel (transparency)
# If the alpha channel is present, it will be the 4th channel in the image array
if image.shape[2] == 4:
    # Use the alpha channel as a mask
    alpha_channel = image[:, :, 3]
    
    # The areas where the mask is not zero, we want to keep, hence we convert it to 0 (black)
    # The areas where the mask is zero, we want to set as 1 (white)
    mask = np.where(alpha_channel == 0, 0, 1).astype(np.uint8)

    # Save the mask to a file
    mask_path = 'masked_output5.png'
    cv2.imwrite(mask_path, mask * 255)  # We multiply by 255 to convert from 0,1 to 0,255

    mask_path
else:
    print("The image does not have an alpha channel or is not in the correct format.")