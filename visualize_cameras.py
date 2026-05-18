#!/usr/bin/env python3
"""
Visualize camera poses from transforms_train/test.json alongside the GLB mesh.
Usage:
    python visualize_cameras.py
Then open http://localhost:8080 in your browser.
"""

import json
import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation
import sys

GLB_PATH = sys.argv[1]
TRAIN_JSON = f"{sys.argv[2]}/transforms_train.json"
TEST_JSON = f"{sys.argv[2]}/transforms_test.json"


def load_cameras(json_path):
    with open(json_path) as f:
        data = json.load(f)
    fovx = float(data["camera_angle_x"])
    c2ws = [np.array(frame["transform_matrix"]) for frame in data["frames"]]
    return fovx, c2ws


def c2w_to_wxyz_pos(c2w: np.ndarray):
    """Extract viser-compatible (w,x,y,z) quaternion and position from a C2W matrix.

    transforms_train.json uses OpenGL convention: camera looks in -Z, +Z is backward.
    Viser's frustum opens in the camera's +Z, so we flip Y and Z (the standard
    OpenGL→OpenCV conversion) to make +Z point forward into the scene.
    """
    pos = c2w[:3, 3]
    flip_yz = np.diag([1.0, -1.0, -1.0])
    rot = Rotation.from_matrix(c2w[:3, :3] @ flip_yz)
    xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    return wxyz, pos


def main():
    with open(GLB_PATH, "rb") as f:
        glb_bytes = f.read()

    fovx_train, train_c2ws = load_cameras(TRAIN_JSON)
    fovx_test, test_c2ws = load_cameras(TEST_JSON)

    server = viser.ViserServer(host="0.0.0.0", port=8080, label="Camera Viewer")
    server.set_up_direction("+y")

    # Mesh and cameras are both in world coordinates — no transform needed.
    server.add_glb("mesh/table", glb_bytes)

    # GUI toggles
    gui_train = server.add_gui_checkbox("Show train cameras", initial_value=True)
    gui_test = server.add_gui_checkbox("Show test cameras", initial_value=True)

    train_handles = []
    for i, c2w in enumerate(train_c2ws):
        wxyz, pos = c2w_to_wxyz_pos(c2w)
        h = server.add_camera_frustum(
            f"cameras/train/{i:04d}",
            fov=fovx_train,
            aspect=1.0,
            scale=0.08,
            color=(50, 200, 80),
            wxyz=wxyz,
            position=pos,
        )
        train_handles.append(h)

    test_handles = []
    for i, c2w in enumerate(test_c2ws):
        wxyz, pos = c2w_to_wxyz_pos(c2w)
        h = server.add_camera_frustum(
            f"cameras/test/{i:04d}",
            fov=fovx_test,
            aspect=1.0,
            scale=0.08,
            color=(220, 60, 60),
            wxyz=wxyz,
            position=pos,
        )
        test_handles.append(h)

    print(f"Train cameras: {len(train_handles)} (green)")
    print(f"Test cameras:  {len(test_handles)} (red)")
    print("Open http://localhost:8080 in your browser")

    @gui_train.on_update
    def _toggle_train(_):
        for h in train_handles:
            h.visible = gui_train.value

    @gui_test.on_update
    def _toggle_test(_):
        for h in test_handles:
            h.visible = gui_test.value

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
