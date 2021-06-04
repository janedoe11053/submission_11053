import math

import torch

class ImpactMeasure(object):

    def __init__(self, behavior, baseline, model, loss=None, *, features_shape=None):
        self.behavior = behavior
        self.baseline = baseline
        self.model = model
        self.loss = loss
        self._features_shape = features_shape
        self._device_type = None

    @classmethod
    def build(cls, behavior, baseline, model, loss=None, *, features_shape=None):
        return cls(behavior, baseline, model, loss=loss, features_shape=features_shape)

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        self._behavior = behavior

    @property
    def baseline(self):
        return self._baseline

    @baseline.setter
    def baseline(self, baseline):
        self._baseline = baseline

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def features_shape(self):
        return self._features_shape

    @property
    def device_type(self):
        return self._device_type

    def infer_features_shape(self):
        if self.features_shape is None:
            for behavior_sample, _ in self.behavior:
                return behavior_sample.shape[1:]
        return self.features_shape

    def generate_mask(self, subset_repr, shape=None):
        if shape is None:
            shape = self.infer_features_shape()
        n_features = math.prod(shape)
        indices = torch.tensor(subset_repr, dtype=torch.long, device=self.device_type)
        mask = torch.zeros(n_features, dtype=torch.bool, device=self.device_type)
        mask[indices] = True
        return mask.view(*shape)

    def __call__(self, mask, shape=None):
        if isinstance(mask, (tuple, list)):
            mask = self.generate_mask(mask, shape)

        for behavior_sample, behavior_targets in self.behavior:
            n_behaviors = behavior_sample.shape[0]
            impact_measure_val = None
            baseline_count = 0
            for baseline_sample, _ in self.baseline:
                assert behavior_sample.dim() == baseline_sample.dim()
                assert behavior_sample.shape[1:] == baseline_sample.shape[1:]
                assert behavior_sample.device == baseline_sample.device
                assert behavior_sample.dtype == baseline_sample.dtype
                if self.features_shape is None:
                    self._features_shape = behavior_sample.shape[1:]
                else:
                    assert self.features_shape == behavior_sample.shape[1:]
                if self.device_type is None:
                    self._device_type = behavior_sample.device
                else:
                    assert self.device_type == behavior_sample.device

                n_baselines = baseline_sample.shape[0]
                masked_input = (behavior_sample.unsqueeze(1) * mask
                                + baseline_sample.unsqueeze(0) * ~mask)
                predictions = self.model(masked_input.view(-1, *masked_input.shape[2:]))
                if self.loss is None:
                    random_vars = predictions
                else:
                    targets = behavior_targets.repeat_interleave(
                        predictions.shape[0] // behavior_targets.shape[0],
                        dim=0
                    )
                    random_vars = self.loss(predictions, targets)
                random_vars_reshaped = random_vars.view(n_behaviors, n_baselines, *random_vars.shape[1:])
                if impact_measure_val is not None:
                    impact_measure_val += random_vars_reshaped.sum(1)
                else:
                    impact_measure_val = random_vars_reshaped.sum(1)
                baseline_count += baseline_sample.shape[0]
            impact_measure_val /= baseline_count
            yield impact_measure_val

if __name__ == '__main__':
    features_space_size = 10
    def model():
        weights = torch.rand(features_space_size)
        def val(x):
            return (x * weights).sum(-1, keepdim=True)
        return val

    torch.manual_seed(13)
    behavior = (
        (torch.rand(3, features_space_size), torch.rand(3, 1)),
        (torch.rand(3, features_space_size), torch.rand(3, 1)),
    )
    baseline = (
        (torch.rand(3, features_space_size), torch.rand(3, 1)),
        (torch.rand(3, features_space_size), torch.rand(3, 1)),
    )
    im = ImpactMeasure(behavior, baseline, model=model())
    for val in im([0, 2, 4]):
        print(val)
