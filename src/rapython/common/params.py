class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}(" + ", ".join(f"{key}={value}" for key, value in self.__dict__.items()) + ")"

    def __repr__(self):
        return self.__str__()


class TrainParams(Params):
    def __init__(self, input_type, alpha=0.03, beta=0.1, constraints_rate=0.3, is_partial_list=True, **kwargs):
        super().__init__(**kwargs)
        self.input_type = input_type
        self.alpha = alpha
        self.beta = beta
        self.constraints_rate = constraints_rate
        self.is_partial_list = is_partial_list


class TestParams(Params):
    def __init__(self, using_average_w=True, **kwargs):
        super().__init__(**kwargs)
        self.using_average_w = using_average_w


def function1(a: Params):
    if isinstance(a, TrainParams):
        train_params = a  # 直接转化
    else:
        raise TypeError("Expected a TrainParams instance")
    print(train_params.alpha)
