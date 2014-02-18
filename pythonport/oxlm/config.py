class ModelData(object):
    """Class to hold model data."""
    def __init__(self, step_size=0.1, l2_parameter=0.0, l1_parameter=0.0, source_l2_parameter=0.0, threads=1, iteration_size=1, verbose=False, ngram_order=3, word_representation_size=100, classes=1, nonlinear=False, diagonal=False, source_window_width=-1, source_eos=True):
        super(ModelData, self).__init__()

        self.step_size = step_size
        self.l2_parameter = l2_parameter
        self.l1_parameter = l1_parameter
        self.source_l2_parameter = source_l2_parameter
        self.threads = threads
        self.iteration_size = iteration_size
        self.verbose = verbose
        self.ngram_order = ngram_order
        self.word_representation_size = word_representation_size
        self.classes = classes
        self.nonlinear = nonlinear
        self.diagonal = diagonal
        self.source_window_width = source_window_width
        self.source_eos = source_eos