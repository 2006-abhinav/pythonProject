import cv2
import numpy as np

image_path = "uas takimages/1.png"
original_img = cv2.imread(image_path)
hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

green_lower_hsv = np.array([36, 108, 0])
green_upper_hsv = np.array([117, 255, 255])
green_mask = cv2.inRange(hsv_img, green_lower_hsv, green_upper_hsv)

cyan_overlay = np.full_like(original_img, [255, 255, 102], dtype=np.uint8)
masked_green = cv2.bitwise_and(cyan_overlay, cyan_overlay, mask=green_mask)

burnt_lower_hsv = np.array([1, 110, 0])
burnt_upper_hsv = np.array([36, 255, 255])
burnt_mask = cv2.inRange(hsv_img, burnt_lower_hsv, burnt_upper_hsv)

yellow_overlay = np.full_like(original_img, [102, 255, 255], dtype=np.uint8)
masked_burnt = cv2.bitwise_and(yellow_overlay, yellow_overlay, mask=burnt_mask)

overall_mask = green_mask | burnt_mask
inverted_mask = cv2.bitwise_not(overall_mask)

preserved_original = cv2.bitwise_and(original_img, original_img, mask=inverted_mask)

final_output = cv2.addWeighted(masked_green, 0.6, masked_burnt, 0.4, 0)
final_output = cv2.bitwise_or(final_output, preserved_original)

cv2.imshow("Processed Image", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
