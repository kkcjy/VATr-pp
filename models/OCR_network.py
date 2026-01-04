import torch
from .networks import *


class BiLSTM(nn.Module):

    def __init__(self, n_in, n_hid, n_out):
        super(BiLSTM, self).__init__()

        self.rnn = nn.LSTM(n_in, n_hid, bidirectional=True)
        self.emb = nn.Linear(n_hid * 2, n_out)


    def forward(self, x):
        rec, _ = self.rnn(x)
        T, b, h = rec.size()
        t_rec = rec.view(T * b, h)

        out = self.emb(t_rec)  # [T * b, n_out]
        out = out.view(T, b, -1)

        return out


class CRNN(nn.Module):

    def __init__(self, args, leaky=False):
        super(CRNN, self).__init__()
        self.args = args
        self.name = 'OCR'
        self.add_noise = False
        self.noise = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([0.2]))

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()
        nh = 256
        handle_nan = False

        def conv_relu(i, bn=False):
            n_in = 1 if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if bn:
                cnn.add_module('bn{0}'.format(i), nn.BatchNorm2d(n_out))
            if leaky:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        conv_relu(0)
        cnn.add_module('pool{0}'.format(0), nn.MaxPool2d(2, 2))
        conv_relu(1)
        cnn.add_module('pool{0}'.format(1), nn.MaxPool2d(2, 2))
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module('pool{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(4, True)
        if self.args.resolution==63:
            cnn.add_module('pool{0}'.format(3),
                           nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(5)
        cnn.add_module('pool{0}'.format(4),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(6, True)

        self.cnn = cnn
        self.use_rnn = False
        if self.use_rnn:
            self.rnn = nn.Sequential(
                BiLSTM(512, nh, nh),
                BiLSTM(nh, nh, ))
        else:
            self.lin = nn.Linear(512, self.args.vocab_size)

        if handle_nan:
            self.register_backward_hook(self.back_hook)

        self.dev = torch.device('cuda:{}'.format(0))
        self.init = 'N02'
        
        self = init_weights(self, self.init)

    def forward(self, x):
        if self.add_noise:
            x = x + self.noise.sample(x.size()).squeeze(-1).to(self.args.device)
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        if h!=1:
            print('a')
        assert h == 1
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        if self.use_rnn:
            out = self.rnn(conv)
        else:
            out = self.lin(conv)
        return out

    def back_hook(self, mod, grad_in, grad_out):
        for g in grad_in:
            g[g != g] = 0


class LabelConv:
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alph (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alph, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alph = alph.lower()
        self.alph = alph + '-'

        self.dict = {}
        for i, ch in enumerate(alph):
            self.dict[ch] = i + 1

    def encode(self, texts):
        """Support batch or single str.
        Args:
            texts (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [len_0 + len_1 + ... len_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        lens = []
        res = []
        all_res = []
        for txt in texts:
            if isinstance(txt, bytes): txt = txt.decode('utf-8', 'strict')
            lens.append(len(txt))
            for ch in txt:
                idx = self.dict[ch]
                res.append(idx)
            all_res.append(res)
            res = []

        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(txt) for txt in all_res], batch_first=True), torch.IntTensor(lens), None

    def decode(self, t, lens, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [len_0 + len_1 + ... len_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if lens.numel() == 1:
            l = lens[0]
            assert t.numel() == l
            if raw:
                return ''.join([self.alph[i - 1] for i in t])
            else:
                chars = []
                for i in range(l):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        chars.append(self.alph[t[i] - 1])
                return ''.join(chars)
        else:
            assert t.numel() == lens.sum()
            txts = []
            idx = 0
            for i in range(lens.numel()):
                l = lens[i]
                txts.append(
                    self.decode(
                        t[idx:idx + l], torch.IntTensor([l]), raw=raw))
                idx += l
            return txts