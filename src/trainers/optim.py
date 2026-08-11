# Copyright 2022 Microsoft Corporation.
'''
Optimizers with μP scaling.

Here we provide 3 ready-to-go optimizers MuAdam, MuAdamW, and MuSGD.
However, the user can easily convert their own optimizer to a μP
optimizer: if your `optimizer` is "Adam-like", such as RMSProp and Adagrad,
that involves normalizing the gradient entrywise, then the following creates
the desired μP optimizer:

    def MuOptimizer(params, **kwargs):
        return MuAdam(params, impl=optimizer, **kwargs)

On the other hand, if your `optimizer` is "SGD-like", such as ASGD, then
the following creates the desired μP optimizer:

    def MuOptimizer(params, **kwargs):
        return MuSGD(params, impl=optimizer, **kwargs)

See Appendix B in our paper for discussions of other optimizers.
'''
from collections import defaultdict
from torch.optim import SGD, Adam, AdamW
from models.utils import depth_mup_parameter_scales


def process_param_groups(params, **kwargs):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]
    for param_group in param_groups:
        if 'lr' not in param_group:
            param_group['lr'] = kwargs['lr']
        if 'weight_decay' not in param_group:
            param_group['weight_decay'] = kwargs.get('weight_decay', 0.)
    return param_groups


def MuAdam(params, impl = Adam, decoupled_wd = False, **kwargs):
    '''Adam with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.

    Inputs:
        impl: the specific Adam-like optimizer implementation from torch.optim or
            elsewhere
        decoupled_wd: if True, skips the mup scaling for weight decay, which should
            be used for optimizer implementations that decouple weight decay from
            learning rate. See https://github.com/microsoft/mup/issues/1 for a use case.
    Outputs:
        An instance of `impl` with refined parameter groups, each of which has the correctly
        scaled learning rate according to mup.
    '''
    param_groups = process_param_groups(params, **kwargs)
    depth_mup_enabled = any(getattr(p, '_depth_mup_enabled', False) for param_group in param_groups for p in param_group['params'])

    new_param_groups = []
    for param_group in param_groups:
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        matrix_like_p = defaultdict(new_group)  # key is width/depth scaling
        vector_like_p = defaultdict(new_group) if depth_mup_enabled else new_group()
        for p in param_group['params']:
            assert hasattr(p, 'infshape'), (f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                                            'Did you forget to call `mup.set_base_shapes` on the model?')
            if p.infshape.ninf() == 2:
                width_mult = p.infshape.width_mult()
                lr_scale, eps_scale = depth_mup_parameter_scales(p)
                matrix_like_p[(width_mult, lr_scale, eps_scale)]['params'].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError('more than 2 inf dimensions')
            else:
                if depth_mup_enabled:
                    lr_scale, eps_scale = depth_mup_parameter_scales(p)
                    vector_like_p[(lr_scale, eps_scale)]['params'].append(p)
                else:
                    vector_like_p['params'].append(p)
        for (width_mult, lr_scale, eps_scale), group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            group['lr'] *= lr_scale / width_mult
            if not decoupled_wd:
                group['weight_decay'] *= width_mult
            if depth_mup_enabled:
                group['eps'] = group.get('eps', kwargs.get('eps', 1e-8)) * eps_scale

        if depth_mup_enabled:
            for (lr_scale, eps_scale), group in vector_like_p.items():
                group['lr'] *= lr_scale
                group['eps'] = group.get('eps', kwargs.get('eps', 1e-8)) * eps_scale
            vector_groups = list(vector_like_p.values())
        else:
            vector_groups = [vector_like_p]

        new_param_groups.extend(list(matrix_like_p.values()) + vector_groups)
    return impl(new_param_groups, **kwargs)


def MuAdamW(params, **kwargs):
    '''AdamW with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.
    '''
    return MuAdam(params, impl = AdamW, **kwargs)


def MuSGD(params, impl = SGD, decoupled_wd = False, **kwargs):
    '''SGD with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.

    Inputs:
        impl: the specific SGD-like optimizer implementation from torch.optim or
            elsewhere
        decoupled_wd: if True, skips the mup scaling for weight decay, which should
            be used for optimizer implementations that decouple weight decay from
            learning rate. See https://github.com/microsoft/mup/issues/1 for a use case.
    Outputs:
        An instance of `impl` with refined parameter groups, each of which has the correctly
        scaled learning rate according to mup.
    '''
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        vector_like_p = defaultdict(new_group)  # key is width mult
        matrix_like_p = defaultdict(new_group)  # key is fan_in/out ratio
        fixed_p = new_group()
        for p in param_group['params']:
            assert hasattr(p, 'infshape'), (f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                                            'Did you forget to call `mup.set_base_shapes` on the model?')
            if p.infshape.ninf() == 1:
                vector_like_p[p.infshape.width_mult()]['params'].append(p)
            elif p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.fanin_fanout_mult_ratio()]['params'].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError('more than 2 inf dimensions')
            else:
                fixed_p['params'].append(p)
        for width_mult, group in vector_like_p.items():
            # Scale learning rate and weight decay accordingly
            group['lr'] *= width_mult
            if not decoupled_wd:
                group['weight_decay'] /= width_mult
        for shape_ratio, group in matrix_like_p.items():
            group['lr'] /= shape_ratio
            if not decoupled_wd:
                group['weight_decay'] *= shape_ratio
        new_param_groups.extend(list(matrix_like_p.values()) + \
                                list(vector_like_p.values()) + [fixed_p])
    return impl(new_param_groups, **kwargs)
