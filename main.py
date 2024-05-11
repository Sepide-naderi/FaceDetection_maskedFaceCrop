import cv2
import mediapipe
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import pandas as pd

# Create a Tkinter window
root = tk.Tk()
# Hide the main window
root.withdraw()

file_path = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.jfif"),
                                                                             ("All files", "*.*")))
# Load the image
image = cv2.imread(file_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
results = face_mesh.process(image[:,:,::-1])

landmarks = results.multi_face_landmarks[0]

face_oval = mp_face_mesh.FACEMESH_FACE_OVAL


df = pd.DataFrame(list(face_oval), columns= ['p1', 'p2'])

routes_index = []

p1 = df.iloc[0]['p1']
p2 = df.iloc[0]['p2']

for i in range(0, df.shape[0]):
    obj = df[df['p1'] == p2]
    p1 = obj['p1'].values[0]
    p2 = obj['p2'].values[0]

    curr_route = []
    curr_route.append(p1)
    curr_route.append(p2)
    routes_index.append(curr_route)


for route_index in routes_index:
    print(f'draw a line from {routes_index[0]} landmark point to {routes_index[1]} landmark point')

routes = []
for src_index, target_index in routes_index:
    source = landmarks.landmark[src_index]
    target = landmarks.landmark[target_index]

    relative_source = int(source.x * image.shape[1]), int(source.y * image.shape[0])
    relative_target = int(target.x * image.shape[1]), int(target.y * image.shape[0])

    routes.append(relative_source)
    routes.append(relative_target)


mask = np.zeros((image.shape[0], image.shape[1]))
mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
mask = mask.astype(bool)

out = np.zeros_like(image)
out[mask] = image[mask]

directory = os.path.dirname('Outputs/')
filename = os.path.basename(file_path)

# Generate a new filename
new_filename = "croppedFace_" + filename

# Save the masked image
cv2.imwrite(os.path.join(directory, new_filename), out[:,:,::-1])

# Display the masked image
cv2.imshow("Masked Face", out[:,:,::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
