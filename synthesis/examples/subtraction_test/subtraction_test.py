import cv2 
import numpy as np
import argparse
import os

postcontrast_image_ = cv2.imread("Breast_MRI_001_0001_slice37-1.png");
precontrast_image_ = cv2.imread("Breast_MRI_001_0001_slice37-2.png");
print("post:")
print(postcontrast_image_)
print("pre:")
print(precontrast_image_)

subtracted_image = cv2.subtract(postcontrast_image_, precontrast_image_)

print("subtracted:")
print(subtracted_image)

cv2.imwrite("Breast_MRI_001_0001_slice37_subtracted-test.png", subtracted_image)