"""Dataset kwargs for Open X-Embodiment datasets.

Target configuration:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB
    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth
    proprio_encoding: Type of proprio encoding used
    action_encoding: Type of action encoding used, e.g. EEF position vs joint position control
"""

from __future__ import annotations

from enum import IntEnum


class ProprioEncoding(IntEnum):
    """Defines supported proprio encoding schemes for different datasets."""

    NONE = -1  # no proprio provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + gripper open/close
    POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    JOINT = 3  # joint angles + gripper open/close
    JOINT_BIMANUAL = 4  # 2 x [6 x joint angles + gripper open/close]
    POS_NAV = 5  # XY + yaw
    QUADRUPED = 6

    # single arm
    MANO = 7  # Hand 3D Pose (21 x 3) + MANO Pose (16 x 3) + cam (3x3) = 120 total


class ActionEncoding(IntEnum):
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = 1  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    JOINT_POS = 2  # 7 x joint delta position + gripper open/close
    JOINT = JOINT_POS  # alias
    JOINT_POS_BIMANUAL = 3  # 2 x [6 x joint pos + gripper]
    NAV_2D = 4  # [delta_x, delta_y] waypoint
    JOINT_POS_BIMANUAL_NAV = 5  # 2 x [6 x joint pos + gripper] + linear base vel + angular base vel
    QUADRUPED = 6

    # single arm
    MANO = 7  # Hand 3D Pose (21 x 3) + MANO Pose (16 x 3) + cam (3x3) = 120 total


class ActionDim(IntEnum):
    # OTHER
    NAV_2D = 2
    JOINT_POS = 8
    JOINT_POS_BIMANUAL_NAV = 14
    QUADRUPED = 12

    # MAIN
    SINGLE = 7
    BIMANUAL = 14
    MANO = 120

    # DEBUG
    DMANO_6 = 6  # xyz&rot
    DMANO_7 = 7  # xyz&rot grip
    # DMANO_18 = 18  # 3 palm & 15 finger params
    DMANO_35 = 35  # xyz,rot, 6*3 major knuckles and thumb , 11*1 other knuckles
    DMANO_51 = 51  # 3 palm & 48 pose params
    DMANO_52 = 52  #
    DMANO_XYZ = 63

    K3DS = 84  # 21 joints x 4 ... xyz+conf?


class ProprioDim(IntEnum):
    POS_EULER = 7
    POS_QUAT = 8
    JOINT = 8
    BIMANUAL = 14
    POS_NAV = 3
    QUADRUPED = 46

    # +1 to account for focal length which is needed for perspective mat
    DMANO_6 = ActionDim.DMANO_6 + 1
    DMANO_7 = ActionDim.DMANO_7 + 1
    DMANO_51 = ActionDim.DMANO_51 + 1
    DMANO_52 = ActionDim.DMANO_52 + 1

    MANO = DMANO_7  # current setting


# clean up data spec
proprio = {}


class PreDict:
    # _TEMPLATE = {}

    def __init__(self, keys):
        """Generate a predefined dictionary with optional overrides."""

        # self._TEMPLATE.copy()
        self.TEMPLATE = dict.fromkeys(keys)

    def __call__(self, **kwargs):
        """Makes the class callable, delegating to the `generate` method."""
        return self.generate(**kwargs)

    def generate(self, **kwargs):
        """Generate a dictionary with optional overrides."""

        out = self.TEMPLATE.copy()
        for key, value in kwargs.items():
            if key in out:
                out[key] = value
            else:
                raise KeyError(f"'{key}' is not a valid key in the template.")
        return out


IMOBS = PreDict(["primary", "high", "side", "nav", "left_wrist", "right_wrist"])
DIMOBS = PreDict(["primary", "secondary", "wrist"])
POBS = PreDict(["single", "mano", "bimanual", "quadruped"])

xgym = {
    "image_obs_keys": {
        "primary": "worm",
        # "high": "overhead",
        "side": "side",
        "nav": None,
        "left_wrist": "wrist",
        "right_wrist": None,
    },
    "depth_obs_keys": DIMOBS(),
    "state_obs_keys": [],
    "proprio_obs_keys": POBS(single="proprio"),
    "proprio_obs_dims": {
        "mano": ProprioDim.MANO,
        "single": ProprioDim.JOINT,
        "bimanual": ProprioDim.BIMANUAL,
        "quadruped": ProprioDim.QUADRUPED,
    },
    "proprio_encoding": ProprioEncoding.JOINT,  # roll-pitch-yaw + gripper open/close
    "action_encoding": ActionEncoding.JOINT_POS,  # EEF_POS,
}

#
# TODO use some sort of metaclass to handle this cleanly?
# it is a very complicated setup

mano = {
    "image_obs_keys": IMOBS(primary="image"),
    "depth_obs_keys": DIMOBS(),
    "state_obs_keys": [
        "cam_intr",
        "mano_pose",
        "mano_shape",
        "joints_3d",
        "joints_vis",
    ],
    "proprio_obs_keys": POBS(mano="proprio"),
    # what is significant about this?
    # re: i think its needed for datasets where there is no proprio so they can fill zeros
    "proprio_obs_dims": {
        "mano": ProprioDim.MANO,
        "single": ProprioDim.POS_EULER,
        "bimanual": ProprioDim.BIMANUAL,
        "quadruped": ProprioDim.QUADRUPED,
    },
    "proprio_encoding": ProprioEncoding.MANO,
    "action_encoding": ActionEncoding.MANO,
}

# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "xgym_lift_mano": mano,
    "sweep_mano": mano,
    "xgym_stack_mano": mano,
    "xgym_duck_mano": mano,
    "xgym_lift_single": xgym,
    "xgym_duck_single": xgym,
    "xgym_stack_single": xgym,
    "xgym_sweep_single": xgym,
    "xgym_play_single": xgym,
    "xgym_single": xgym,
    "my_dataset": xgym,
    # "rlds_oakink": mano,  # OAK INK Dataset
    # the rest were removed for now since we dont use that data
}
