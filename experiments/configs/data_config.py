import ml_collections

ACT_MEAN = [
    1.9296819e-04,
    1.3667766e-04,
    -1.4583133e-04,
    -1.8390431e-04,
    -3.0808983e-04,
    2.7425270e-04,
    5.9716219e-01,
]

ACT_STD = [
    0.00912848,
    0.0127196,
    0.01229497,
    0.02606696,
    0.02875283,
    0.07807977,
    0.48710242,
]

ACTION_PROPRIO_METADATA = {
    "action": {
        "mean": ACT_MEAN,
        "std": ACT_STD,
        # TODO compute these
        "min": ACT_MEAN,
        "max": ACT_STD,
    },
    # TODO compute these
    "proprio": {"mean": ACT_MEAN, "std": ACT_STD, "min": ACT_MEAN, "max": ACT_STD},
}


def get_config(config_string):
    possible_structures = {
        "all": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        # "icra/?*/?*/?*",
                        # "flap/?*/?*/?*",
                        # "bridge_data_v1/berkeley/?*/?*",
                        # "rss/?*/?*/?*",
                        # "bridge_data_v2/?*/?*/?*",
                        "scripted/?*",
                        # "bridge_dataset/?*",
                    ]
                ],
                "exclude": [],
                "sample_weights": None,
                "action_proprio_metadata": ACTION_PROPRIO_METADATA,
            }
        )
    }
    return possible_structures[config_string]
