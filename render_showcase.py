"""Render the task scene the way MuJoCo's own demos look, to show that it is a
real 3D scene and not a flat sprite generator.

Nothing here changes the task. The dataset renders with shadows off and an
untextured floor because shadow mapping is what software rendering (llvmpipe,
under WSL) is slow at -- roughly 130 img/s with shadows against 800+ without.
That is a 6x difference on the only cost that matters during dataset generation,
so the dataset stays plain and the pretty version lives in this file.

    python render_showcase.py
"""
import os
import numpy as np
import mujoco

import common as C
import gen_dataset as G

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720

ASSETS = """  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.32 0.45 0.62"
             rgb2="0.04 0.06 0.10" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.28 0.40"
             rgb2="0.26 0.38 0.52" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="30 30" texuniform="true"
              reflectance="0.25"/>
  </asset>
"""


def showcase_model():
    xml = open(os.path.join(ROOT, "scene.xml")).read()
    xml = xml.replace('offwidth="256"', f'offwidth="{W}"').replace('offheight="256"', f'offheight="{H}"')
    xml = xml.replace('shadowsize="0"', 'shadowsize="4096"')
    xml = xml.replace('<worldbody>', ASSETS + "\n  <worldbody>")
    xml = xml.replace('rgba="0.75 0.75 0.72 1"/>', 'material="grid"/>')
    return mujoco.MjModel.from_xml_string(xml)


def free_cam(azimuth, elevation, distance, lookat=(0, 0, 0.03)):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.azimuth, c.elevation, c.distance = azimuth, elevation, distance
    c.lookat[:] = lookat
    return c


def main(seed=3):
    model = showcase_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    rng = np.random.default_rng(seed)

    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(C.N_SLOTS)]
    gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}") for i in range(C.N_SLOTS)]
    light = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "l0")

    pts = G.sample_positions(rng, C.N_SLOTS)
    objs = [G.sample_object(rng) for _ in pts]
    for s, (p, (cls, gtype, size, half_h, yaw)) in enumerate(zip(pts, objs)):
        model.geom_type[gids[s]] = gtype
        model.geom_size[gids[s]] = size
        model.geom_rgba[gids[s]] = [*C.hsv_to_rgb(rng.uniform(0, 1), 0.75, 0.85), 1.0]
        G.place(model, bids[s], p, half_h, yaw)
    model.light_pos[light] = [0.25, -0.2, 1.1]
    model.light_diffuse[light] = [0.85, 0.85, 0.85]
    mujoco.mj_forward(model, data)

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    views = {
        "showcase_orbit": free_cam(135, -22, 0.62),
        "showcase_low": free_cam(200, -8, 0.55),
        "showcase_task_tilt": "tilt",
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, cam in views.items():
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        plt.imsave(os.path.join(ROOT, "out", name + ".png"), img)
        print(f"wrote out/{name}.png")


if __name__ == "__main__":
    main()
