import cv2
import numpy as np

edge_img = cv2.imread('edges_img.jpg', cv2.IMREAD_GRAYSCALE)

mask = np.zeros_like(edge_img)
mask = cv2.fillPoly(mask,
                    np.array([[[558, 368], [85, 368], [313, 213], [335, 212]]]),
                    color=255)

masked_edge_img = cv2.bitwise_and(edge_img, mask)
cv2.imwrite('masked_edge_img.jpg', masked_edge_img)
cv2.imshow('masked', masked_edge_img)
cv2.waitKey(0)


