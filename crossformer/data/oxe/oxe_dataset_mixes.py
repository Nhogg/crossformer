"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""

from __future__ import annotations

HEAD_TO_DATASET = {
    "mano": [
        "sweep_mano",
        "rlds_oakink",
        "xgym_lift_mano",
        "xgym_stack_mano",
        "xgym_duck_mano",
    ],
    # 'binamo': [ ],
    # 'smplx': [ ],
    "nav": ["omnimimic_gnm_dataset"],
    "single_arm": [
        "xgym_lift_single",
        "xgym_stack_single",
        "xgym_duck_single",
        "xgym_sweep_single",
        "my_dataset",
        #
        "berkeley_mvp_converted_externally_to_rlds",
        "nyu_rot_dataset_converted_externally_to_rlds",
        "ucsd_kitchen_dataset_converted_externally_to_rlds",
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        "utokyo_xarm_bimanual_converted_externally_to_rlds",
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
        #
        "bridge_dataset",
        "fractal20220817_data",
        "kuka",
        "taco_play",
        "taco_extra",
        "jaco_play",
        "berkeley_cable_routing",
        "roboturk",
        "nyu_door_opening_surprising_effectiveness",
        "viola",
        "berkeley_autolab_ur5",
        "toto",
        "language_table",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "austin_buds_dataset_converted_externally_to_rlds",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
        "furniture_bench_dataset_converted_externally_to_rlds",
        "austin_sailor_dataset_converted_externally_to_rlds",
        "austin_sirius_dataset_converted_externally_to_rlds",
        "bc_z",
        "dlr_edan_shared_control_converted_externally_to_rlds",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "utaustin_mutex",
        "berkeley_fanuc_manipulation",
        "cmu_stretch",
        "droid",
        "droid_wipe",
        "droid_flip_pot_upright",
    ],
    "bimanual": [
        "aloha_pen_uncap_diverse_dataset",
        "aloha_new_sushi_dataset",
        "aloha_dough_cut_dataset",
        "aloha_lucy_dataset",
        "aloha_drawer_dataset",
        "aloha_pick_place_dataset",
        "aloha_static_dataset",
        "aloha_sushi_cut_full_dataset",
        "aloha_new_sushi_dataset,",
    ],
    "quadruped": ["go1_real_dataset", "a1", "go1"],
}
HEAD_TO_DATASET = {k: HEAD_TO_DATASET[k] for k in ["single_arm", "mano"]}

DATASET_TO_HEAD = {d: head for head, dsets in HEAD_TO_DATASET.items() for d in dsets}

OXE_MAGIC_SOUP_BALANCED = [
    ("kuka", 0.14503701874493363),
    ("taco_play", 0.06657998827701668),
    ("taco_extra", 0.015452958868388737),
    ("jaco_play", 0.010914534155076169),
    ("berkeley_cable_routing", 0.005925612796973822),
    ("roboturk", 0.052499238268860826),
    ("nyu_door_opening_surprising_effectiveness", 0.0028565519070650833),
    ("viola", 0.021369612129854),
    ("berkeley_autolab_ur5", 0.027421498380401588),
    ("toto", 0.045595496181288435),
    ("language_table", 0.09863155061985435),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 0.10030032010542056),
    ("austin_buds_dataset_converted_externally_to_rlds", 0.004775432426062442),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 0.01884652293499813),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.05526993262706029),
    ("austin_sailor_dataset_converted_externally_to_rlds", 0.04943059735717906),
    ("austin_sirius_dataset_converted_externally_to_rlds", 0.03918942829266809),
    ("bc_z", 0.14503701874493363),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 0.00124985520344411),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 0.020472678629801757),
    ("utaustin_mutex", 0.05066099356944051),
    ("berkeley_fanuc_manipulation", 0.017530731149920712),
    ("cmu_stretch", 0.003502058441908362),
    ("droid", 0.001450370187449336),
]

CROSS_EMBODIMENT_TARGET = [
    ("aloha_pen_uncap_diverse_dataset", 0.1),
    ("aloha_new_sushi_dataset", 0.2),
    ("bridge_dataset", 0.2),
    ("a1", 0.1),
    ("droid_wipe", 0.1),
    ("droid_flip_pot_upright", 0.1),
    ("omnimimic_gnm_dataset", 0.2),
]

CROSS_EMBODIMENT = [(name, weight * 0.15) for name, weight in OXE_MAGIC_SOUP_BALANCED] + [
    (name, weight * 0.85) for name, weight in CROSS_EMBODIMENT_TARGET
]


allweight = lambda arr, w: [(name, weight * w) for name, weight in arr]

XARM_EXT = [
    ("bridge_dataset", 0.2),
    ("berkeley_mvp_converted_externally_to_rlds", 0.2),
    ("nyu_rot_dataset_converted_externally_to_rlds", 0.2),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 0.2),
    ("ucsd_pick_and_place_dataset_converted_externally_to_rlds", 0.2),
    # ("utokyo_xarm_bimanual_converted_externally_to_rlds", 0.2),
    # ("utokyo_xarm_pick_and_place_converted_externally_to_rlds", 0.2),
]  # + allweight(OXE_MAGIC_SOUP_BALANCED, 0.01)

XGYM_SINGLE = [
    ("xgym_duck_single", 1.0),
    ("xgym_lift_single", 1.0),
    ("xgym_stack_single", 1.0),
]


XGYM_ALL = [
    ("xgym_duck_single", 1.0),
    ("xgym_lift_single", 1.0),
    ("xgym_stack_single", 1.0),
    #
    ("xgym_lift_mano", 1.0),
]


MANO = [("rlds_oakink", 1.0)]

# BAFL_SOUP = allweight(MANO, 0.25) + allweight(XARM_EXT, 0.25) + allweight(XGYM_ALL, 0.5)

OXE_NAMED_MIXES = {
    "cross_embodiment": CROSS_EMBODIMENT,
    # "bafl": BAFL_SOUP,
    "bridge": [("bridge_dataset", 1.0)],
    #
    "x": XGYM_ALL,
    "xs": XGYM_SINGLE,
    "xstack": [("xgym_stack_single", 1.0), ("xgym_stack_mano", 1.0)],
    "xduck": [("xgym_duck_single", 1.0), ("xgym_duck_mano", 1.0)],
    #
    **{k: [(k, 1.0)] for sublist in HEAD_TO_DATASET.values() for k in sublist},  # o(n)
}
