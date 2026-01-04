import collections

OLD_KEYS = ['netG._logvarD.bias', 'netG._logvarD.weight', 'netG._logvarE.bias', 'netG._logvarE.weight', 'netG._muD.bias', 'netG._muD.weight', 'netG._muE.bias', 'netG._muE.weight', 'netD.embed.weight', 'netD.embed.u0', 'netD.embed.sv0', 'netD.embed.bias']


def load_gen(model, ckpt):
    if not isinstance(ckpt, collections.OrderedDict):
        ckpt = ckpt['model']

    ckpt = {k.replace("netG.",""): v for k, v in ckpt.items() if k.startswith("netG") and k not in OLD_KEYS}
    model.netG.load_state_dict(ckpt)

    return model


def load_ckpt(model, ckpt):
    if not isinstance(ckpt, collections.OrderedDict):
        ckpt = ckpt['model']
    old_dict = model.state_dict()
    if len(ckpt.keys()) == 241:  # default
        cnt = 0
        for k, v in ckpt.items():
            if k in old_dict:
                old_dict[k] = v
                cnt += 1
            elif 'netG.' + k in old_dict:
                old_dict['netG.' + k] = v
                cnt += 1

        c_keys = [k for k in ckpt.keys() if 'Feat_Encoder' in k]
        o_keys = [k for k in old_dict.keys() if 'Feat_Encoder' in k]
        for ck, ok in zip(c_keys, o_keys):
            old_dict[ok] = ckpt[ck]
            cnt += 1
        assert cnt == 241
        ckpt_dict = old_dict
    else:
        ckpt = {k: v for k, v in ckpt.items() if k not in OLD_KEYS}
        assert len(old_dict) == len(ckpt)
        ckpt_dict = {k2: v1 for (k1, v1), (k2, v2) in zip(ckpt.items(), old_dict.items()) if
                           v1.shape == v2.shape}
    assert len(old_dict) == len(ckpt_dict)
    model.load_state_dict(ckpt_dict, strict=False)
    return model