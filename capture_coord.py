import cv2
import numpy as np

# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((8, 2), np.int)

counter = 0


def mousePoints(event, x, y, flags, params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x, y
        counter = (counter + 1) % 8


# Read image
# img = cv2.imread('../data/sudoku-original.jpg')

def capture(img, windows_name):
    while True:
        for x in range(0, 8):
            cv2.circle(img, (point_matrix[x][0], point_matrix[x][1]), 3, (0, 255, 0), cv2.FILLED)

        # Showing original image
        cv2.imshow(windows_name, img)
        # Mouse click event on original image
        cv2.setMouseCallback(windows_name, mousePoints)
        # Printing updated point matrix
        # print(point_matrix)
        # Refreshing window all time
        k = cv2.waitKey(1) & 0xFF

        # Exit if the user presses the ESC key
        if k == 27:
            break

    # Destroy all windows
    cv2.destroyAllWindows()
    return point_matrix.astype(np.float32)
