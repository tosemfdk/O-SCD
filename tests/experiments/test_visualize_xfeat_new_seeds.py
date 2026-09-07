import numpy as np

from experiments.visualize_xfeat_new_seeds import project_seeds


def test_project_seeds_respects_lifespan_and_projects_initialized_scale():
    camera = {
        "position": [0.0, 0.0, 0.0],
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "fx": 100.0,
        "fy": 100.0,
        "width": 200,
        "height": 100,
    }
    xyz = np.asarray([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]], dtype=np.float32)
    scaling = np.log(np.asarray([[0.03] * 3, [0.03] * 3], dtype=np.float32))
    start = np.asarray([4.0, 6.0], dtype=np.float32)
    end = np.asarray([10.0, np.inf], dtype=np.float32)

    uv, depth, active, radius_px = project_seeds(
        xyz, scaling, start, end, global_index=5, camera=camera
    )

    np.testing.assert_allclose(uv[0], [100.0, 50.0], atol=1e-5)
    np.testing.assert_allclose(uv[1], [110.0, 50.0], atol=1e-5)
    np.testing.assert_allclose(depth, [2.0, 2.0], atol=1e-5)
    np.testing.assert_array_equal(active, [True, False])
    np.testing.assert_allclose(radius_px, [1.5, 1.5], atol=1e-5)
