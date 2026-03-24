from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk  # from : src/pyroki/viewer/_manipulability_ellipse.py
from xgym import calibrate
from xgym.calibrate.urdf.robot import urdf
import yourdfpy


def pose6(q):  # J_6xN = jax.jacfwd(pose6)(joints)     # (6, n)
    T = jaxlie.SE3(robot.forward_kinematics(q))[target_idx]
    p = T.translation()  # (3,)
    r = T.rotation().log()  # (3,) — so(3) local coords (angular)
    return jnp.concatenate([p, r])  # (6,)


urdf = yourdfpy.URDF.load(urdf, mesh_dir=calibrate.urdf.robot.DNAME / "assets")


def atleast_4d(x):
    add = max(0, 4 - x.ndim)
    return x if add == 0 else jnp.expand_dims(x, axis=tuple(range(add)))


def make_robot():
    return pk.Robot.from_urdf(urdf)


def get_jac_fn(robot: pk.Robot, pad_gripper: bool = False):
    names = robot.links.names

    # _q = jnp.zeros((7,))  # 7 DOF

    # x = jaxlie.SE3(robot.forward_kinematics(_q)).translation()
    # kin = {name: _x for name, _x in zip(names, x[0])} # 1,18,7 xyz,qwqxqyqz
    # idx = names.index('link_eef')

    def make_jac_fn(link="link_eef"):
        """returns jacobian of a certain link"""

        def jac_fn(q):
            pads = [(0, 0)] * (q.ndim - 1) + [(0, 1)]
            q = jnp.pad(q, pads, constant_values=0.0) if pad_gripper else q
            shape, q = q.shape, q.reshape(-1, q.shape[-1])

            do_jac = jax.jacfwd(lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation())
            x = jax.vmap(do_jac)(q)  # jacobian only operates on vector

            x = x[..., names.index(link), :, :]  # select link
            x = x.reshape(shape[:-1] + x.shape[-2:])
            x = x[..., :7] if pad_gripper else x
            return x

        return jac_fn

    jac_fn = make_jac_fn("link_eef")
    # jacobian = jac_fn(_q)
    # assert jacobian.shape == (3, robot.joints.num_actuated_joints)
    return jac_fn


def get_fwd_kin_fn(
    robot: pk.Robot,
    pad_gripper: bool = False,
):
    names = robot.links.names

    def make_fwd_kin_fn(link="link_eef"):
        """returns forward kinematics of a certain link"""

        def fwd_kin_fn(q):
            pads = [(0, 0)] * (q.ndim - 1) + [(0, 1)]
            q = jnp.pad(q, pads, constant_values=0.0) if pad_gripper else q
            return jaxlie.SE3(robot.forward_kinematics(q)).translation()[..., names.index(link), :]

        return fwd_kin_fn

    return make_fwd_kin_fn("link_eef")


def test():
    for shape in [(7,), (1, 7), (10, 7), (2, 3, 7), (4, 5, 6, 7)]:
        q = jnp.zeros(shape)
        x = fwd_kin_fn(q)
        xd = jac_fn(q)
        assert x.shape == (*shape[:-1], 3)
        assert xd.shape == (*shape[:-1], 3, 7)
