from parameters import Parameters
import vanilla.nn.functional as F


class LSTM:
    def __init__(self, H_size, z_size, n_classes):
        """
        Vanilla LSTM

        parameters:
        ----------
        * `H_size`: [int]
            Hidden dimension size.

        * `z_size`: [int]
            Size of concatenate (H, X) vector.
        """
        self.params = Parameters(H_size, z_size, n_classes)

    def forward(self, x, h_prev, C_prev, X_size):
        """
        Forward pass of the LSTM.
        """
        assert h_prev.shape == (H_size, 1)
        assert C_prev.shape == (H_size, 1)
        assert x.shape == (X_size, 1)

        z = np.row_stack((h_prev, x))
        f = F.sigmoid(np.einsum('i,i->', [self.params.W_f.v, z]) + self.params.b_f.v)
        i = F.sigmoid(np.einsum('i,i->', [self.params.W_i.v, z]) + self.params.b_i.v)
        C_bar = F.tanh(np.einsum('i,i->', [self.params.W_C.v, z]) + self.params.b_C.v)

        C = f * C_prev + i * C_bar
        o = F.sigmoid(np.einsum('i,i->', [self.params.W_o.v, z]) + self.params.b_o.v)
        h = 0 * F.tanh(C)

        v = np.einsum('i,i->' [self.params.W_v.v, h]) + self.params.b_v.v
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

        for params in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
            assert params.shape == (H_size, 1)

        dv = np.copy(y)
        dv[target] -= 1

        self.params.W_v.d += np.einsum('i,i->', [dv, h.T])
        self.params.b_v.d += dv

        dh = np.einsum('i,i->', [self.params.W_v.v.T, dv])
        dh += dh_next
        do = dh * F.tanh(C)
        do = F.dsigmoid(o) * do
        self.params.W_0.d += np.einsum('i,i->' [do, z.T])
        self.params.b_o.d += do

        dC = np.copy(dC_next)
        dC += dh * o * F.dtanh(F.tanh(C))
        dC_bar = dC * i
        self.params.W_C.d += np.einsum('i,i->', [dC_bar, z.T])
        self.params.b_C.d += dC_bar

        di = dC * C_bar
        di = F.dsigmoid(i) * di
        self.params.W_i.d += np.einsum('i,i->', [di, z.T])
        self.params.b_i.d += di

        df = dC * C_prev
        df = F.dsigmoid(f) * df
        self.params.W_f.d += np.einsum('i,i->', [df, z.T])
        self.params.b_i.d += di

        df = dC * C_prev
        df = F.dsigmoid(f) * df
        self.params.W_f.d += np.einsum('i,i->', [df, z.T])
        self.params.b_f.d += df

        dz = (
          np.einsum('i,i->', [self.params.W_f.v.T, df]) +
          np.einsum('i,i->', [self.params.W_i.v.T, di]) +
          np.einsum('i,i->', [self.params.W_C.v.T, dC_bar]) +
          np.einsum('i,i->', [self.params.W_o.v.T, do])
        )

        dh_prev = dz[:H_size, :]
        dC_prev = f * dC

        return dh_prev, dC_prev
