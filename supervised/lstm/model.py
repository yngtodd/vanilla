from parameters import Parameters


class LSTM:
    def __init__(self, H_size, z_size, n_classes):
        """
        Vanilla LSTM

        Parameters:
        ----------
        * `H_size`: [int]
            Hidden dimension size.

        * `z_size`: [int]
            Size of concatenate (H, X) vector.
        """
        self.parameters = Parameters(H_size, z_size, n_classes)
