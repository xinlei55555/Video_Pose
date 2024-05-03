import cv2

# Load the infrared image
infrared_image = cv2.imread('infrared_image.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(infrared_image, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray_image, 100, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Create a blank image of normal colors
normal_colors_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Fill the outline with a specified color (e.g., skin color)
cv2.drawContours(normal_colors_image, [largest_contour], -1, (255, 255, 204), cv2.FILLED)

# Save the result to a new file
cv2.imwrite('outlined_person.jpg', normal_colors_image)

# Display the result
cv2.imshow('Outline of Person', normal_colors_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
