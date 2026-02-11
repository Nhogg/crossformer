from __future__ import annotations

from enum import Enum


class ActionSpace(Enum):
    NONE = 0
    QUADRUPED = 12
    NAV = 2  # 2D

    # single action space
    POS_EULER = 7  # EEF XYZ + roll-pitch-yaw + gripper open/close
    POS_QUAT = 8  # EEF XYZ + quaternion + gripper open/close
    JOINT = 8  # joint angles + gripper open/close

    # bimanual action space
    BI_POS_EULER = 14  # 2x(6+1)
    BI_POS_QUAT = 16  # 2x(7+1)
    BI_JOINT = 16  # 2x(7+1)

    # human action
    # TODO make better names
    # DMANO_18 = 18  # 3 palm & 15 finger params
    DMANO_7 = 7  # xyz,rot,grippiness
    DMANO_35 = 35  # xyz,rot, 6*3 major knuckles and thumb , 11*1 other knuckles
    DMANO_51 = 51  # 3 palm & 48 pose params
    DMANO_52 = 52  #

    MANO = 63  # 21 joints x 3
    K3DS = 84  # 21 joints x 4 ... xyz+conf?


class ActionRep(Enum):
    """Action representation: relative or absolute"""

    RELATIVE = "rel"
    ABSOLUTE = "abs"


class Head(Enum):
    """Output head / Action Space of the model"""

    BIMANUAL = "bimanual"
    QUADRUPED = "quadruped"
    NAV = "nav"
    SINGLE = "single"
    SINGLE_ARM = "single"  # remap. first definition overrides
    MANO = "mano"
    K3DS = "k3ds"

    MULTI = "multi"  # reserved for MultiDataSource


HEAD2SPACE = {  # convert head name to action space int
    Head.BIMANUAL: ActionSpace.BI_POS_EULER,
    Head.QUADRUPED: ActionSpace.QUADRUPED,
    Head.NAV: ActionSpace.NAV,
    Head.SINGLE: ActionSpace.JOINT,
    Head.MANO: ActionSpace.MANO,
    Head.MANO: ActionSpace.MANO,
    Head.K3DS: ActionSpace.K3DS,
}
