import cv2
import numpy as np

# create a blank white image
img = 255 * np.ones((300, 300, 3), dtype=np.uint8)

# draw a red circle
cv2.circle(img, (150, 150), 80, (0, 0, 255), -1)

# show window
cv2.imshow("Test Window", img)

print("Press any key in the window to close it...")
cv2.waitKey(0)
cv2.destroyAllWindows()
