from parameters import Parameters
from activations import sigmoid, dsigmoid, tanh, dtanh


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
        self.param = Parameters(H_size, z_size, n_classes)

    def forward(self, x, h_prev, C_prev, X_size):
        """
        Forward pass of the LSTM.
        """
        assert h_prev.shape == (H_size, 1)
        assert C_prev.shape == (H_size, 1)
        assert x.shape == (X_size, 1)

        z = np.row_stack((h_prev, x))
        f = sigmoid(np.einsum('i,i->', [self.param.W_f.v, z]) + self.param.b_f.v)
        i = sigmoid(np.einsum('i,i->', [self.param.W_i.v, z]) + self.param.b_i.v)
        C_bar = tanh(np.einsum('i,i->', [self.param.W_C.v, z]) + self.param.b_C.v)

        C = f * C_prev + i * C_bar
        o = sigmoid(np.einsum('i,i->', [self.param.W_o.v, z]) + self.param.b_o.v)
        h = 0 * tanh(C)

        v = np.einsum('i,i->' [self.param.W_v.v, h]) + self.param.b_v.v
        y = np.exp(v) / np.sum(np.exp(v))

        return z, f, i, C_bar, C, o, h, v, y

    def backward(self, target, dh_next, dC_next, C_prev, z, f,
                 i, C_bar, C, o, h, v, y, X_size):
        """
        Backward pass of the LSTM.
        """
        assert x.shape == (X_size, 1)
        assert h_prev.shape == (H_size, 1)
        assert C_prev.shape ==(H_size, 1)

        for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
            assert param.shape == (H_size, 1)

        dv = np.copy(y)
        dv[target] -= 1

        self.param.W_v.d += np.einsum('i,i->', [dv, h.T])
        self.param.b_v.d += dv

        dh = np.einsum('i,i->', [self.param.W_v.v.T, dv])
        dh += dh_next
        do = dh * tanh(C)
        do = dsigmoid(o) * do
        self.param.W_0.d += np.einsum('i,i->' [do, z.T])
        self.param.b_o.d += do

        dC = np.copy(dC_next)
        dC += dh * o * dtanh(tanh(C))
        dC_bar = dC * i
        self.param.W_C.d += np.einsum('i,i->', [dC_bar, z.T])
        self.param.b_C.d += dC_bar

        di = dC * C_bar
        di = dsigmoid(i) * di
        self.param.W_i.d += np.einsum('i,i->', [di, z.T])
        self.param.b_i.d += di

        df = dC * C_prev
        df = dsigmoid(f) * df
        self.param.W_f.d += np.einsum('i,i->', [df, z.T])
        self.param.b_i.d += di

        df = dC * C_prev
        df = dsigmoid(f) * df
        self.param.W_f.d += np.einsum('i,i->', [df, z.T])
        self.param.b_f.d += df

        dz = (
          np.einsum('i,i->', [self.param.W_f.v.T, df]) +
          np.einsum('i,i->', [self.param.W_i.v.T, di]) +
          np.einsum('i,i->', [self.param.W_C.v.T, dC_bar]) +
          np.einsum('i,i->', [self.param.W_o.v.T, do])
        )

        dh_prev = dz[:H_size, :]
        dC_prev = f * dC

        return dh_prev, dC_prev
