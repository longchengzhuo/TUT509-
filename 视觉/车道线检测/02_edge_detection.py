import cv2

img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

edge_img = cv2.Canny(img, 50, 100)

cv2.imwrite('edges_img.jpg', edge_img)
cv2.imshow('edges', edge_img)
cv2.waitKey(0)