#!/usr/bin/env python3
"""
Render multi-view RGBA images from a textured GLB mesh using nvdiffrast.
Outputs a NeRF/Blender-format dataset for training 2DGS.

Usage:
    python render_mesh_multiview.py meshes/table.glb --output data/table
    python train.py -s data/table --init_mesh meshes/table.glb --white_background
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from PIL import Image
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def mesh_bounding_sphere(vertices: np.ndarray):
    """Return (center, radius) of the AABB-midpoint bounding sphere."""
    center = (vertices.max(0) + vertices.min(0)) / 2.0
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    return center, max(radius, 1e-8)


def look_at_opengl(eye: np.ndarray, center: np.ndarray = None, up: np.ndarray = None) -> np.ndarray:
    """Return 4×4 C2W matrix in OpenGL convention (camera looks in -Z, +Y up)."""
    if center is None:
        center = np.zeros(3)
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    eye = np.asarray(eye, dtype=np.float64)
    f = center - eye
    norm_f = np.linalg.norm(f)
    if norm_f < 1e-8:
        f = np.array([0.0, 0.0, -1.0])
    else:
        f = f / norm_f
    r = np.cross(f, up / np.linalg.norm(up))
    if np.linalg.norm(r) < 1e-8:
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, 0] = r
    C2W[:3, 1] = u
    C2W[:3, 2] = -f   # camera looks in −Z
    C2W[:3, 3] = eye
    return C2W


def perspective_gl(fovy_deg: float, aspect: float, near: float = 0.01, far: float = 100.0) -> np.ndarray:
    """Standard OpenGL perspective projection matrix (column-vector convention)."""
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array([
        [f / aspect, 0,  0,                          0                       ],
        [0,          f,  0,                          0                       ],
        [0,          0,  (far + near) / (near - far), 2 * far * near / (near - far)],
        [0,          0, -1,                          0                       ],
    ], dtype=np.float32)


def generate_sphere_cameras(n_cameras: int = 128,
                            center: np.ndarray = None,
                            radius: float = 2.5) -> list:
    """Return list of 4×4 C2W matrices (OpenGL) uniformly distributed on a sphere.

    Cameras are placed on a sphere of `radius` centred at `center` (world coords)
    and all look toward `center`.  Uses the Fibonacci / golden-angle spiral for
    uniform angular density.
    """
    if center is None:
        center = np.zeros(3)
    golden_ratio = (1.0 + 5.0 ** 0.5) / 2.0
    cameras = []
    for i in range(n_cameras):
        cos_theta = 1.0 - 2.0 * (i + 0.5) / n_cameras
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta ** 2))
        phi = 2.0 * np.pi * i / golden_ratio
        direction = np.array([sin_theta * np.cos(phi),
                               cos_theta,
                               sin_theta * np.sin(phi)])
        eye = center + radius * direction
        look = -direction
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(look, up)) > 0.9:
            up = np.array([0.0, 0.0, 1.0])
        cameras.append(look_at_opengl(eye, center=center, up=up))
    return cameras


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_glb(path: str) -> list:
    """
    Load a GLB file and return a list of submesh dicts:
      {'vertices': (N,3) f32, 'faces': (F,3) i32,
       'uvs': (N,2) f32 | None, 'texture': PIL.Image | None,
       'vertex_colors': (N,3) f32 | None}
    Vertices are in world space (scene graph transforms applied).
    """
    scene = trimesh.load(path, force='scene')
    submeshes = []
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name].copy()
        geom.apply_transform(transform)

        verts = np.array(geom.vertices, dtype=np.float32)
        faces = np.array(geom.faces, dtype=np.int32)

        uvs = None
        texture_img = None
        vertex_colors = None

        if isinstance(geom.visual, trimesh.visual.color.ColorVisuals):
            # Direct per-vertex RGBA stored in the GLB
            vc = np.array(geom.visual.vertex_colors, dtype=np.float32)  # (N,4) uint8
            vertex_colors = vc[:, :3] / 255.0
            print(f"  [{geom_name}] vertex colors: {vertex_colors.shape}")

        elif isinstance(geom.visual, trimesh.visual.texture.TextureVisuals):
            try:
                uvs = np.array(geom.visual.uv, dtype=np.float32)
                mat = geom.visual.material
                img = getattr(mat, 'image', None) or getattr(mat, 'baseColorTexture', None)
                if img is not None:
                    texture_img = img.convert('RGB')
                    print(f"  [{geom_name}] UV texture: {texture_img.size}")
                else:
                    uvs = None
            except Exception as e:
                print(f"  [{geom_name}] UV/texture failed ({e}), falling back to material color")
                uvs = None

            if uvs is None:
                # Material base color → constant vertex color
                try:
                    factor = getattr(geom.visual.material, 'baseColorFactor', None)
                    if factor is not None:
                        c = np.array(factor[:3], dtype=np.float32)
                    else:
                        c = np.array([0.8, 0.8, 0.8], dtype=np.float32)
                    vertex_colors = np.tile(c, (len(verts), 1))
                except Exception:
                    vertex_colors = np.full((len(verts), 3), 0.8, dtype=np.float32)

        else:
            vertex_colors = np.full((len(verts), 3), 0.8, dtype=np.float32)

        submeshes.append({
            'vertices': verts,
            'faces': faces,
            'uvs': uvs,
            'texture': texture_img,
            'vertex_colors': vertex_colors,
        })
    return submeshes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _make_texture_tensor(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Convert a PIL RGB image to an nvdiffrast-ready texture tensor (1, H, W, 3) f32.
    The image is flipped vertically so that UV v=0 maps to the bottom (OpenGL convention).
    """
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = np.flipud(arr)                   # PIL row-0 = top; GL v=0 = bottom
    return torch.tensor(arr, device=device).unsqueeze(0)  # (1, H, W, 3)


def render_submesh(glctx, sub: dict, W2C: np.ndarray, proj: np.ndarray,
                   H: int, W: int, device: torch.device):
    """Render one submesh. Returns (H, W, 3) rgb and (H, W, 1) alpha float tensors."""
    verts = torch.tensor(sub['vertices'], dtype=torch.float32, device=device)  # (N,3)
    faces = torch.tensor(sub['faces'],    dtype=torch.int32,   device=device)  # (F,3)

    # Clip-space positions: pos = MVP @ [x,y,z,1]^T
    ones = torch.ones(verts.shape[0], 1, device=device)
    verts_h = torch.cat([verts, ones], dim=1)                      # (N,4)
    MVP = torch.tensor(proj @ W2C, dtype=torch.float32, device=device)  # (4,4)
    pos = (MVP @ verts_h.T).T.contiguous().unsqueeze(0)             # (1,N,4)

    rast, _ = dr.rasterize(glctx, pos.contiguous(), faces.contiguous(), resolution=[H, W])  # (1,H,W,4)
    alpha = (rast[..., 3:4] > 0).float()                                                   # (1,H,W,1)

    if sub['uvs'] is not None and sub['texture'] is not None:
        uvs = torch.tensor(sub['uvs'], dtype=torch.float32, device=device).contiguous()  # (N,2)
        uv_map, _ = dr.interpolate(uvs.unsqueeze(0).contiguous(), rast, faces)           # (1,H,W,2)
        tex_t = _make_texture_tensor(sub['texture'], device)                              # (1,H_t,W_t,3)
        rgb = dr.texture(tex_t.contiguous(), uv_map.contiguous())                        # (1,H,W,3)
    else:
        vc = torch.tensor(sub['vertex_colors'], dtype=torch.float32, device=device).contiguous()  # (N,3)
        rgb, _ = dr.interpolate(vc.unsqueeze(0).contiguous(), rast, faces)                        # (1,H,W,3)

    rgb = dr.antialias(rgb.contiguous(), rast, pos.contiguous(), faces)  # (1,H,W,3)
    return rgb[0], alpha[0]                      # (H,W,3), (H,W,1)


def render_all_views(submeshes: list, c2w_list: list, fovy_deg: float,
                     W: int, H: int, bg_color=(1.0, 1.0, 1.0)) -> list:
    """Render every C2W pose. Returns list of RGBA PIL images."""
    device = torch.device('cuda')
    glctx  = dr.RasterizeCudaContext()
    proj   = perspective_gl(fovy_deg, W / H)

    images = []
    for idx, c2w in enumerate(c2w_list):
        sys.stdout.write(f'\r  Rendering {idx+1}/{len(c2w_list)}')
        sys.stdout.flush()

        W2C = np.linalg.inv(c2w).astype(np.float32)
        rgb_acc   = torch.zeros(H, W, 3, device=device)
        alpha_acc = torch.zeros(H, W, 1, device=device)

        for sub in submeshes:
            rgb, alpha = render_submesh(glctx, sub, W2C, proj, H, W, device)
            # Front-to-back compositing (opaque surfaces — order matters only for overlaps)
            rgb_acc   = rgb_acc   + (1.0 - alpha_acc) * alpha * rgb
            alpha_acc = alpha_acc + (1.0 - alpha_acc) * alpha

        bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
        
        # Extract un-multiplied RGB for clean transparency
        rgb_unmult = torch.where(alpha_acc > 0, rgb_acc / alpha_acc, bg)

        rgba = torch.cat([rgb_unmult, alpha_acc], dim=-1)  # (H,W,4)
        rgba_np = (rgba.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        
        # NVDiffrast outputs with Y=0 at bottom. PIL expects Y=0 at top. We MUST flip it!
        rgba_np = np.flipud(rgba_np)
        images.append(Image.fromarray(rgba_np, 'RGBA'))

    print()
    return images


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------

def export_dataset(images: list, c2w_list: list, fovy_deg: float,
                   output_dir: str, test_every: int = 8):
    """Write images + transforms_{train,test}.json in NeRF/Blender format."""
    train_dir = os.path.join(output_dir, 'train')
    test_dir  = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    # camera_angle_x = horizontal FOV (radians) for a square sensor
    fovx_rad = np.radians(fovy_deg)

    train_frames, test_frames = [], []
    for i, (img, c2w) in enumerate(zip(images, c2w_list)):
        is_test = (i % test_every == 0)
        split   = 'test' if is_test else 'train'
        fname   = f'r_{i:04d}'
        img.save(os.path.join(output_dir, split, fname + '.png'))
        frame = {'file_path': f'{split}/{fname}', 'transform_matrix': c2w.tolist()}
        (test_frames if is_test else train_frames).append(frame)

    for frames, name in [
        (train_frames, 'transforms_train.json'),
        (test_frames,  'transforms_test.json'),
    ]:
        with open(os.path.join(output_dir, name), 'w') as fp:
            json.dump({'camera_angle_x': fovx_rad, 'frames': frames}, fp, indent=2)

    print(f"Saved {len(train_frames)} train + {len(test_frames)} test frames → {output_dir}")


def export_point_cloud(submeshes: list, output_path: str):
    """
    Write face centroids as points3d.ply with per-face colors.
    The Blender loader picks this up automatically when present.
    """
    all_xyz, all_rgb = [], []
    for sub in submeshes:
        verts = sub['vertices']
        faces = sub['faces']
        centroids = verts[faces].mean(axis=1)          # (F,3)
        all_xyz.append(centroids)

        if sub['uvs'] is not None and sub['texture'] is not None:
            uvs = sub['uvs']                            # (N,2)
            cuv  = uvs[faces].mean(axis=1)             # (F,2) centroid UV
            tex  = np.array(sub['texture'])            # (H,W,3) uint8
            th, tw = tex.shape[:2]
            px = (cuv[:, 0]        * (tw - 1)).astype(int).clip(0, tw - 1)
            py = ((1 - cuv[:, 1]) * (th - 1)).astype(int).clip(0, th - 1)
            all_rgb.append(tex[py, px, :3])            # (F,3) uint8
        elif sub['vertex_colors'] is not None:
            vc = sub['vertex_colors']                  # (N,3) float
            fc = vc[faces].mean(axis=1)                # (F,3) float
            all_rgb.append((fc * 255).astype(np.uint8))
        else:
            n = len(centroids)
            all_rgb.append(np.full((n, 3), 200, dtype=np.uint8))

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    dtype = [('x','f4'),('y','f4'),('z','f4'),
             ('nx','f4'),('ny','f4'),('nz','f4'),
             ('red','u1'),('green','u1'),('blue','u1')]
    nrm  = np.zeros_like(xyz)
    elem = np.empty(len(xyz), dtype=dtype)
    for i, row in enumerate(np.concatenate([xyz, nrm, rgb.astype(np.float32)], axis=1)):
        elem[i] = tuple(row)
    PlyData([PlyElement.describe(elem, 'vertex')]).write(output_path)
    print(f"Saved {len(xyz)}-point cloud → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render multi-view images from a textured GLB mesh")
    parser.add_argument('mesh', help='Path to .glb mesh file')
    parser.add_argument('--output', '-o', default='data/mesh_dataset')
    parser.add_argument('--n_cameras', type=int, default=128,
                        help='Total number of cameras (Fibonacci sphere sampling)')
    parser.add_argument('--distance_factor', type=float, default=2.5,
                        help='Camera distance as a multiple of the mesh bounding radius')
    parser.add_argument('--fovy', type=float, default=50.0, help='Vertical FOV in degrees')
    parser.add_argument('--width',  type=int, default=800)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--black_bg', action='store_true', default=False)
    parser.add_argument('--test_every', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading mesh: {args.mesh}")
    submeshes = load_glb(args.mesh)
    print(f"  Loaded {len(submeshes)} submesh(es): "
          + ", ".join(f"{len(s['faces'])}f" for s in submeshes))

    all_verts = np.concatenate([s['vertices'] for s in submeshes], axis=0)
    mesh_center, bounding_radius = mesh_bounding_sphere(all_verts)
    camera_radius = bounding_radius * args.distance_factor
    print(f"  Bounding sphere: center={mesh_center.round(4)}, radius={bounding_radius:.4f}")
    print(f"  Camera radius:   {camera_radius:.4f} ({args.distance_factor}x bounding radius)")

    print("Generating camera poses...")
    c2w_list = generate_sphere_cameras(n_cameras=args.n_cameras,
                                       center=mesh_center,
                                       radius=camera_radius)
    print(f"  Total cameras: {len(c2w_list)}")

    bg = (0.0, 0.0, 0.0) if args.black_bg else (1.0, 1.0, 1.0)
    print("Rendering views...")
    images = render_all_views(submeshes, c2w_list, fovy_deg=args.fovy,
                              W=args.width, H=args.height, bg_color=bg)

    print("Exporting dataset...")
    export_dataset(images, c2w_list, args.fovy, args.output, test_every=args.test_every)

    ply_path = os.path.join(args.output, 'points3d.ply')
    export_point_cloud(submeshes, ply_path)

    print("\nDone! To train 2DGS with mesh initialization:")
    print(f"  python train.py -s {args.output} --init_mesh {args.mesh} --white_background")


if __name__ == '__main__':
    main()
