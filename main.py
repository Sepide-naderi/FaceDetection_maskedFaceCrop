import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt

# Create a Tkinter window
root = tk.Tk()
# Hide the main window
root.withdraw()

# Ask the user to select an image file for their face
user_face_path = filedialog.askopenfilename(title="Select Your Face Image",
                                            filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.jfif"),
                                                       ("All files", "*.*")))

# Ask the user to select an image file for the skin
skin_face_path = filedialog.askopenfilename(title="Select Skin Face Image",
                                            filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.jfif"),
                                                       ("All files", "*.*")))

# Load the images
user_face_image = cv2.imread(user_face_path)
skin_face_image = cv2.imread(skin_face_path)

user_face_image_rgb = cv2.cvtColor(user_face_image, cv2.COLOR_BGR2RGB)
skin_face_image_rgb = cv2.cvtColor(skin_face_image, cv2.COLOR_BGR2RGB)

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Process the images to get landmarks
user_face_results = face_mesh.process(user_face_image_rgb)
skin_face_results = face_mesh.process(skin_face_image_rgb)

# If no faces are detected in either image, exit the program
if not user_face_results.multi_face_landmarks or not skin_face_results.multi_face_landmarks:
    print("No face detected in one or both of the images.")
    exit()

user_face_landmarks = user_face_results.multi_face_landmarks[0].landmark
skin_face_landmarks = skin_face_results.multi_face_landmarks[0].landmark

# Convert landmarks to numpy array
def landmarks_to_array(landmarks, image_shape):
    return np.array([(int(landmark.x * image_shape[1]),
                      int(landmark.y * image_shape[0]))
                     for landmark in landmarks])

user_face_points = landmarks_to_array(user_face_landmarks, user_face_image.shape)
skin_face_points = landmarks_to_array(skin_face_landmarks, skin_face_image.shape)

#Delaunay triangulation for the skin face
def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for point in points:
        point_int = (int(point[0]), int(point[1]))
        subdiv.insert(point_int)
    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []
    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if (rect[0] <= pt1[0] <= rect[0] + rect[2] and
                rect[1] <= pt1[1] <= rect[1] + rect[3] and
                rect[0] <= pt2[0] <= rect[0] + rect[2] and
                rect[1] <= pt2[1] <= rect[1] + rect[3] and
                rect[0] <= pt3[0] <= rect[0] + rect[2] and
                rect[1] <= pt3[1] <= rect[1] + rect[3]):
            index = []
            for j in range(3):
                for k in range(len(points)):
                    if abs(pt1[0] - points[k][0]) < 1 and abs(pt1[1] - points[k][1]) < 1:
                        index.append(k)
                pt1 = pt2
                pt2 = pt3
            if len(index) == 3:
                delaunay_triangles.append((index[0], index[1], index[2]))
    return delaunay_triangles

rect = (0, 0, skin_face_image.shape[1], skin_face_image.shape[0])
delaunay_triangles = calculate_delaunay_triangles(rect, skin_face_points)

# Warp the triangles from user face to skin face
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - x1), (t1[i][1] - y1)))
        t2_rect.append(((t2[i][0] - x2), (t2[i][1] - y2)))
        t2_rect_int.append(((t2[i][0] - x2), (t2[i][1] - y2)))

    img1_rect = img1[y1:y1 + h1, x1:x1 + w1]
    size = (w2, h2)
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    mask = np.zeros((h2, w2, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img2[y2:y2 + h2, x2:x2 + w2] = img2[y2:y2 + h2, x2:x2 + w2] * (1 - mask) + img2_rect * mask

# Create a mask for the face
def create_face_mask(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array([(int(landmark.x * image.shape[1]),
                        int(landmark.y * image.shape[0]))
                       for landmark in landmarks])
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    return mask

# Create masks for user face and skin face
user_face_mask = create_face_mask(user_face_image, user_face_landmarks)
skin_face_mask = create_face_mask(skin_face_image, skin_face_landmarks)

# check if the mask is in the correct shape
if len(skin_face_mask.shape) == 2:
    skin_face_mask = cv2.cvtColor(skin_face_mask, cv2.COLOR_GRAY2BGR)

# Copy triangle
img_skin_face = np.copy(skin_face_image_rgb)
for triangle in delaunay_triangles:
    t1 = [user_face_points[triangle[0]], user_face_points[triangle[1]], user_face_points[triangle[2]]]
    t2 = [skin_face_points[triangle[0]], skin_face_points[triangle[1]], skin_face_points[triangle[2]]]
    warp_triangle(user_face_image_rgb, img_skin_face, t1, t2)

# Feather the edges of the face mask(more ksize->smoother edges)
face_mask = cv2.GaussianBlur(skin_face_mask, (101, 101), 5)

# Find the center of the face in the skin image for seamless cloning
def find_face_center(landmarks):
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    center_x = int(np.mean(x_coords) * skin_face_image.shape[1])
    center_y = int(np.mean(y_coords) * skin_face_image.shape[0])
    return (center_x, center_y)

center_point = find_face_center(skin_face_landmarks)

# Blend the images
result = cv2.seamlessClone(img_skin_face, skin_face_image_rgb, face_mask, center_point, cv2.NORMAL_CLONE)

# Save the final image
directory = os.path.dirname('Outputs/')
os.makedirs(directory, exist_ok=True)
filename = os.path.basename(user_face_path)
output_filename = "cropped_face_" + filename
cv2.imwrite(os.path.join(directory, output_filename), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

# Display the final result
plt.figure(figsize=(10, 10))
plt.imshow(result)
plt.title("Combined Face")
plt.axis('off')
plt.show()

# Clean up temporary image
if os.path.exists(os.path.join('Images', filename)):
    os.remove(os.path.join('Images', filename))

# Display the result in OpenCV window
cv2.imshow("Final Skin Face", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
