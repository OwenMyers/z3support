

class ParamsMixin:

    def __init__(self, **kwargs):
        print(f"Params init for class {super(ParamsMixin, self).__class__.__name__}")
        super(ParamsMixin, self).__init__(**kwargs)


class CVAEDenseOnlyParams:

    def __init__(self, **kwargs):
        print("Warning: No parameter setting for the Dense Only model at this time")
        print("    only connects input to latent space")


class CVAECustomParams:
    """
    Minimal right now but latter we can add all sorts of checks here to
    ensure the model is valid
    """

    def __init__(self, **kwargs):
        self.encoder_strides_list = kwargs['encoder_strides_list']
        self.encoder_filters_list = kwargs['encoder_filters_list']
        self.encoder_kernal_list = kwargs['encoder_kernal_list']

        self.decoder_strides_list = kwargs['decoder_strides_list']
        self.decoder_filters_list = kwargs['decoder_filters_list']
        self.decoder_kernal_list = kwargs['decoder_kernal_list']
        self.latent_dim = kwargs['latent_dim']
        self.use_batch_norm = kwargs['use_batch_norm']
        self.use_dropout = kwargs['use_dropout']