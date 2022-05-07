import os
import numpy as np
import trimesh
import matplotlib.cm as cm
import matplotlib as matplotlib
from tqdm import tqdm
import argparse


def color_map_color(value, cmap_name='bwr', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # rgb = cmap(norm(value))  # will return rgba, we take only first 3 so we get rgb
    rgb = cmap(value)  # will return rgba, we take only first 3 so we get rgb
    return rgb


def get_face_verts(mesh: trimesh.Trimesh):
    """get all triangles: (N, 3, 3)"""
    faces = mesh.faces
    verts = mesh.vertices
    face_verts = np.stack([verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]], axis=1)
    return face_verts


def safe_3Dvector_normalize(vecs):
    norm_ = np.linalg.norm(vecs, axis=-1, keepdims=True)
    mask = norm_[..., 0] > 0
    # vecs[mask] = np.array([0, 0, 1]) # NOTE: assign (0, 0, 1) to zero vector
    # norm_[mask] = 1
    vecs = vecs / norm_
    return vecs, mask


def triangle3Dto2D(triangles):
    ca = triangles[:, 0] - triangles[:, 2]
    cb = triangles[:, 1] - triangles[:, 2]
    z_axis = np.cross(ca, cb)
    z_axis, mask1 = safe_3Dvector_normalize(z_axis)
    x_axis, mask2 = safe_3Dvector_normalize(ca)
    y_axis = np.cross(z_axis, x_axis)
    
    new_coords = triangles - triangles[:, 2:3] # (N, 3, 3)
    new_x = np.sum(new_coords * x_axis[:, np.newaxis], axis=-1)
    new_y = np.sum(new_coords * y_axis[:, np.newaxis], axis=-1)
    new_z = np.sum(new_coords * z_axis[:, np.newaxis], axis=-1)
    new_ = np.stack([new_x, new_y, new_z], axis=-1)
    mask = np.logical_and(mask1, mask2)
    return new_, mask


def compute_jacobian(mesh_ori, mesh_emb):
    assert len(mesh_ori.faces) == len(mesh_emb.faces)
    triangles_ori = get_face_verts(mesh_ori)
    triangles_emb = get_face_verts(mesh_emb)
    # print(triangles_ori.shape, triangles_emb.shape)
    
    # 3d to 2d
    triangles_ori, mask_ori = triangle3Dto2D(triangles_ori)
    triangles_ori = triangles_ori[..., :2]
    triangles_emb = triangles_emb[..., :2] # assume the 3rd dimension is zero for mesh_emb
    # how to deal with invalid triangles

    # compute jacobian
    M_ori = triangles_ori[:, 1:3] - triangles_ori[:, 0:1] # (N, 2, 2)
    M_emb = triangles_emb[:, 1:3] - triangles_emb[:, 0:1] # (N, 2, 2)
    sign1 = np.cross(M_ori[:, 0], M_ori[:, 1])
    sign2 = np.cross(M_emb[:, 0], M_emb[:, 1])
    triangles_ori[sign1 * sign2 < 0, :, 1] *= -1
    M_ori = triangles_ori[:, 1:3] - triangles_ori[:, 0:1] # (N, 2, 2)

    # M_ori = M_ori[mask_ori]
    # M_emb = M_emb[mask_ori]

    J = np.einsum('ijk,ikl->ijl', M_emb, np.linalg.inv(M_ori))
    return J


def area_distortion(jacobian, return_each=False):
    energy = (np.linalg.det(jacobian) - 1) ** 2
    if return_each:
        return energy
    return np.sum(energy)


def angle_distortion(jacobian, return_each=False):
    energy = np.linalg.norm(jacobian + jacobian.transpose(0, 2, 1) - 
                2 * np.trace(jacobian, axis1=1, axis2=2).reshape(-1, 1, 1) * np.identity(2)[np.newaxis].repeat(len(jacobian), axis=0),
            'fro', axis=(1, 2)) ** 2
    if return_each:
        return energy
    return np.sum(energy)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-m', '--method', type=str, default=None)
parser.add_argument('-e', '--energy', type=str, default=None)
# parser.add_argument('--vis', action='store_true')
args = parser.parse_args()


src_dir = 'output'


if args.input is None:
    out_dir = 'eval'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    all_names = sorted([x.split('_')[0] for x in os.listdir(src_dir) if x.endswith('original.obj')])
    for name in tqdm(all_names):
        mesh_ori = trimesh.load(f"{src_dir}/{name}_original.obj", process=False)
        save_path = os.path.join(out_dir, f'eval_{name}.txt')
        f = open(save_path, 'w')
        for tag in ['tutte', 'lscm', 'slim', 'arap']:
            print("#####" * 10, file=f)
            print(tag, file=f)
            mesh_emb = trimesh.load(f"{src_dir}/{name}_{tag}.obj", process=False)
            J = compute_jacobian(mesh_ori, mesh_emb)
            dists = area_distortion(J, return_each=True).round(6)
            avg_, med_, max_, min_ = np.mean(dists), np.median(dists), np.max(dists), np.min(dists)
            print("area distortion:", file=f)
            print(avg_, med_, max_, min_, file=f)
            dists = angle_distortion(J, return_each=True).round(6)
            avg_, med_, max_, min_ = np.mean(dists), np.median(dists), np.max(dists), np.min(dists)
            print("angle distortion:", file=f)
            print(avg_, med_, max_, min_, file=f)
        f.close()
else:
    mesh_ori = trimesh.load(f"{src_dir}/{args.input}_original.obj", process=False)
    mesh_emb = trimesh.load(f"{src_dir}/{args.input}_{args.method}.obj", process=False)
    J = compute_jacobian(mesh_ori, mesh_emb)
    if args.energy == 'area':
        dists = area_distortion(J, return_each=True)
    elif args.energy == 'angle':
        dists = angle_distortion(J, return_each=True)
    else:
        raise NotImplementedError

    avg_, max_, min_ = np.mean(dists).round(6), np.max(dists).round(6), np.min(dists).round(6)
    print(avg_, max_, min_)

    dists = np.clip(dists, np.percentile(dists, 10), np.percentile(dists, 90))
    dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists) + 1e-10)
    colors = color_map_color(dists)
    mesh_ori.show()
    mesh_emb.visual.face_colors = colors
    mesh_emb.show()
