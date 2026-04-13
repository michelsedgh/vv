class OutputType:
    SUBWORD = "subword"
    ID = "id"


class BPE:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "youtokentome is not installed in this workspace runtime. "
            "The NeMo speaker benchmark should not need it."
        )
