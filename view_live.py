"""Open the task scene in MuJoCo's interactive viewer -- orbit, zoom, inspect.

Run this from an interactive terminal, not from a script: it opens a window and
blocks until you close it. Under WSL the window comes from WSLg and the rendering
is done on the CPU (llvmpipe), so expect it to be smooth but not fast.

    python view_live.py

Left-drag orbits, right-drag pans, scroll zooms. Press Tab for the control panel.
The 'top' and 'tilt' task cameras are in the camera dropdown, so you can see
exactly what the dataset sees.
"""
import numpy as np
import mujoco
import mujoco.viewer

import common as C
import gen_dataset as G
import render_showcase as S


def main(seed=3, pretty=True):
    model = S.showcase_model() if pretty else G.make_model(192)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)

    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(C.N_SLOTS)]
    gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}") for i in range(C.N_SLOTS)]
    pts = G.sample_positions(rng, C.N_SLOTS)
    for s, p in enumerate(pts):
        cls, gtype, size, half_h, yaw = G.sample_object(rng)
        model.geom_type[gids[s]] = gtype
        model.geom_size[gids[s]] = size
        model.geom_rgba[gids[s]] = [*C.hsv_to_rgb(rng.uniform(0, 1), 0.75, 0.85), 1.0]
        G.place(model, bids[s], p, half_h, yaw)
    mujoco.mj_forward(model, data)

    print("cameras available in the viewer dropdown: top (dataset overhead), "
          "tilt (the detection view)")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
