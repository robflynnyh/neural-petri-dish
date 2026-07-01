import numpy as np
import torch


# Rank-1 directions are normalized to unit RMS, so this is the approximate
# per-wave perturbation RMS added to the shared base weights.
FACTORED_WAVE_COEFF_SCALE = 0.3
NEIGHBOR_INPUT_DIM = 24
HIDDEN_DIM = 64
OUTPUT_DIM = 9
NETWORK_INPUT_DIM = NEIGHBOR_INPUT_DIM + HIDDEN_DIM


def clone_tensor(tensor):
    return tensor.detach().clone()


class LinearGenes:
    def __init__(self, in_features=None, out_features=None, weight=None, bias=None, clone_weight=True, clone_bias=True):
        if weight is None:
            self.weight = torch.empty(out_features, in_features)
            self.bias = torch.empty(out_features)
            torch.nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            bound = in_features ** -0.5
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.weight = clone_tensor(weight) if clone_weight else weight
            self.bias = clone_tensor(bias) if clone_bias else bias
        self.refresh_numpy_views()

    def refresh_numpy_views(self):
        self.weight_np = self.weight.numpy()
        self.bias_np = self.bias.numpy()


def normalized_rank1_factors(out_features, in_features):
    left = torch.randn(out_features)
    right = torch.randn(in_features)
    rms = torch.outer(left, right).square().mean().sqrt().clamp_min(torch.finfo(left.dtype).eps)
    scale = rms.sqrt()
    return left / scale, right / scale


class SharedRank1Family:
    _next_id = 1

    def __init__(self, genes=None):
        if genes is None:
            linear = LinearGenes(NETWORK_INPUT_DIM, HIDDEN_DIM)
            linear2 = LinearGenes(HIDDEN_DIM, OUTPUT_DIM)
            genes = {
                'weight_1': linear.weight,
                'bias_1': linear.bias,
                'weight_2': linear2.weight,
                'bias_2': linear2.bias,
            }
        self.id = SharedRank1Family._next_id
        SharedRank1Family._next_id += 1
        self.base_weight_1 = clone_tensor(genes['weight_1'])
        self.base_bias_1 = clone_tensor(genes['bias_1'])
        self.base_weight_2 = clone_tensor(genes['weight_2'])
        self.base_bias_2 = clone_tensor(genes['bias_2'])
        self.u_1, self.v_1 = normalized_rank1_factors(*self.base_weight_1.shape)
        self.u_2, self.v_2 = normalized_rank1_factors(*self.base_weight_2.shape)
        self.direction_weight_1 = torch.outer(self.u_1, self.v_1)
        self.direction_weight_2 = torch.outer(self.u_2, self.v_2)
        self._device_cache = {}
        self.refresh_numpy_views()

    def refresh_numpy_views(self):
        self.base_weight_1_np = self.base_weight_1.numpy()
        self.base_weight_2_np = self.base_weight_2.numpy()
        self.u_1_np = self.u_1.numpy()
        self.v_1_np = self.v_1.numpy()
        self.u_2_np = self.u_2.numpy()
        self.v_2_np = self.v_2.numpy()
        self.direction_weight_1_np = self.direction_weight_1.numpy()
        self.direction_weight_2_np = self.direction_weight_2.numpy()

    def materialize_weight_1(self, coeff):
        return self.base_weight_1 + self.direction_weight_1 * float(coeff)

    def materialize_weight_2(self, coeff):
        return self.base_weight_2 + self.direction_weight_2 * float(coeff)

    def tensors(self, device):
        device = torch.device(device)
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = {
                'base_weight_1': self.base_weight_1.to(device),
                'base_weight_2': self.base_weight_2.to(device),
                'u_1': self.u_1.to(device),
                'v_1': self.v_1.to(device),
                'u_2': self.u_2.to(device),
                'v_2': self.v_2.to(device),
            }
            self._device_cache[key] = cached
        return cached


def factored_genes(family, coeff_1=0.0, coeff_2=0.0, bias_1=None, bias_2=None):
    if bias_1 is None:
        bias_1 = family.base_bias_1
    if bias_2 is None:
        bias_2 = family.base_bias_2
    return {
        'weight_1': family.materialize_weight_1(coeff_1),
        'bias_1': clone_tensor(bias_1),
        'weight_2': family.materialize_weight_2(coeff_2),
        'bias_2': clone_tensor(bias_2),
        '_rank1_family': family,
        '_rank1_coeff_1': float(coeff_1),
        '_rank1_coeff_2': float(coeff_2),
        '_clone_weight_1': False,
        '_clone_weight_2': False,
    }


def factored_gene_batch(family, count, coeff_scale=FACTORED_WAVE_COEFF_SCALE):
    weight_1, weight_2, coeff_1, coeff_2 = factored_gene_tensors(family, count, coeff_scale)
    return [
        {
            'weight_1': weight_1[index],
            'bias_1': family.base_bias_1,
            'weight_2': weight_2[index],
            'bias_2': family.base_bias_2,
            '_rank1_family': family,
            '_rank1_coeff_1': float(coeff_1[index]),
            '_rank1_coeff_2': float(coeff_2[index]),
            '_clone_weight_1': False,
            '_clone_weight_2': False,
            '_clone_bias_1': False,
            '_clone_bias_2': False,
        }
        for index in range(count)
    ]


def factored_gene_tensors(family, count, coeff_scale=FACTORED_WAVE_COEFF_SCALE):
    coeffs = np.random.randn(count, 2)
    coeff_1 = torch.as_tensor(coeffs[:, 0], dtype=family.base_weight_1.dtype) * coeff_scale
    coeff_2 = torch.as_tensor(coeffs[:, 1], dtype=family.base_weight_2.dtype) * coeff_scale
    weight_1 = family.base_weight_1.unsqueeze(0) + coeff_1.reshape(-1, 1, 1) * family.direction_weight_1.unsqueeze(0)
    weight_2 = family.base_weight_2.unsqueeze(0) + coeff_2.reshape(-1, 1, 1) * family.direction_weight_2.unsqueeze(0)
    return weight_1, weight_2, coeff_1, coeff_2
