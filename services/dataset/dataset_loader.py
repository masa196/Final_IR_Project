import ir_datasets


def load_lotte():
    return ir_datasets.load(
        "lotte/recreation/dev/search"
    )


