

class ParamsMixin:

    def __init__(self, **kwargs):
        print(f"Params init for class {super(ParamsMixin, self).__class__.__name__}")
        self.kwargs_in_dict = kwargs

    def get_hp_dict_for_aim(self):
        d = {}
        for k, v in self.kwargs_in_dict.items():
            d[f"hp_{k}"] = v

        d["hp_model_name"] = self.__class__.__name__

        return d


class CVAEDenseOnlyParams(ParamsMixin):

    def __init__(self, **kwargs):
        super(CVAEDenseOnlyParams, self).__init__(**kwargs)
        self.input_edge_length = kwargs['input_edge_length']


class CVAECustomParams(ParamsMixin):
    """
    Minimal right now but latter we can add all sorts of checks here to
    ensure the model is valid
    """

    def __init__(self, **kwargs):
        super(CVAECustomParams, self).__init__(**kwargs)
        self.input_edge_length = kwargs['input_edge_length']
        self.encoder_strides_list = kwargs['encoder_strides_list']
        self.encoder_filters_list = kwargs['encoder_filters_list']
        self.encoder_kernal_list = kwargs['encoder_kernal_list']

        self.decoder_strides_list = kwargs['decoder_strides_list']
        self.decoder_filters_list = kwargs['decoder_filters_list']
        self.decoder_kernal_list = kwargs['decoder_kernal_list']