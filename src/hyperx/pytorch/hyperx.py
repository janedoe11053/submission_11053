import math
import itertools
import operator

import torch

def compute_mean(generator, sum_f):
    mean = []
    count = 1
    sample_type = None
    for sample_idx, sample in enumerate(generator):
        sample_type = type(sample)
        if sample_type not in (tuple, list):
            sample = (sample,)

        # TODO: insert asserts/tests such that
        # sample[i].shape[0] = sample[j].shape[0] for all i,j in range(len(sample))
        local_count = sample[0].shape[0]
        for subsample_idx, subsample in enumerate(sample):
            local_sum = sum_f(subsample, dim=0)
            if sample_idx == 0:
                mean.append(local_sum / local_count)
            else:
                mean[subsample_idx] = (mean[subsample_idx]
                                       + local_sum / count) * (count / (count + local_count))
        if sample_idx == 0:
            count = local_count
        else:
            count += local_count

    if sample_type in (tuple, list):
        return sample_type(mean)
    else:
        return mean[0]

def map_collections(f, *collections):
    collection_type = type(collections[0])
    # NOTE: comment for now, document this behavior better
    # assert all(map(lambda collection: type(collection) == collection_type, collections))

    if collection_type in (tuple, list):
        return collection_type((
            f(*vals) for vals in zip(*collections)
        ))
    else:
        return f(*collections)

class HyperXExplainer(object):

    def explain(self, impact_measure, feature_subset,
                lowest_s_card=0, highest_s_card=2,  # defines |S| in the half interval [lowest_s_card, highest_s_card)
                centrality='eigenvector',
                store_graph=True,
                theta_map=None,
                omega='uniform',  # now only two possible values are available: 'uniform' or 'binomial'
                p=0,  # passed into `theta_map`, hints a p-norm for the default `theta_map`
                # parameters below are relevant for the power method
                score_min_norm_threshold=1e-5,  # defines a threshold for the norm of a score to be considered as all zeros
                residual_tol=1e-5,  # defines a threshold for the norm proximity stopping criterion in the power method
                niters=30,
                p_normalization=1,  # defines a p-norm in the normalization step of the power method
                *args, **kwargs):

        if lowest_s_card < 0:
            raise ValueError(f"""{type(self)}.explain(): input parameter `lowest_s_card` has to be >= 0""")
        if highest_s_card < 1:
            raise ValueError(f"""{type(self)}.explain(): input parameter `highest_s_card` has to be >= 1""")
        if highest_s_card <= lowest_s_card:
            raise ValueError(f"""{type(self)}.explain(): it must hold that `highest_s_card` > `lowest_s_card`""")

        if omega not in ('uniform', 'binomial'):
            raise ValueError(f"""{type(self)}.explain(): input parameter `omega` has to be in ('uniform', 'binomial')""")

        features_shape = impact_measure.infer_features_shape()
        n_features = math.prod(features_shape)
        if not hasattr(feature_subset, '__len__'):
            feature_subset = list(feature_subset)

        def mean_inducing_theta(v, s, s_card, p):
            s_mask = impact_measure.generate_mask(s, shape=v.shape)
            if s_card == 0:
                return v.norm(p=1)
            if p == 0:
                return (v * s_mask + torch.ones_like(v) * ~s_mask).prod() ** (1 / s_card)
            return (v * s_mask).norm(p=p) / math.pow(s_card, 1 / p)

        if theta_map is None:
            theta_map = mean_inducing_theta
        # decide whether to apply the omega scale.
        # Current options are `uniform`, which is just 1,
        # and `binomial`, which is omega(s, k) = [s choose k]^{-1}
        if omega == 'binomial':
            def omega_inverse_binomial_scale(f):
                def inner(v, s, s_card, p):
                    omega = 1 / math.comb(len(feature_subset), s_card)
                    return omega * f(v, s, s_card, p)
                return inner
            theta_map = omega_inverse_binomial_scale(theta_map)

        def increment_score(score, contribution):
            if score is None:
                return contribution
            else:
                return map_collections(operator.add, score, contribution)

        def expand_as(input, ref):
            if type(ref) in (list, tuple) and type(input) not in (list, tuple):
                return type(ref)((input,) * len(ref))
            else:
                return input

        class FEvaluator(object):
            def __init__(self, explainer, store_graph=False):
                if store_graph:
                    if not hasattr(explainer, 'graph'):
                        explainer.graph = {}
                    self.graph = explainer.graph
                self.store_graph = store_graph

            def __call__(self, x_pos, x_neg):
                scores_pos = torch.zeros_like(x_pos)
                scores_neg = torch.zeros_like(x_neg)
                for feature_idx, feature in enumerate(feature_subset):
                    feature_pos_score = None
                    feature_neg_score = None
                    for s_card in range(lowest_s_card, highest_s_card):
                        s_feature_subsets_generator = itertools.combinations(feature_subset, s_card)
                        s_x_subsets_generator = itertools.combinations(range(len(feature_subset)), s_card)
                        for s, s_x in zip(s_feature_subsets_generator, s_x_subsets_generator):
                            if feature in s:
                                continue

                            s_feature_edge = (s, feature)
                            if not self.store_graph or s_feature_edge not in self.graph:
                                mask_with_feature = impact_measure.generate_mask(s + (feature,), shape=features_shape)
                                mask_no_feature = impact_measure.generate_mask(s, shape=features_shape)
                                impact_with_feature = compute_mean(impact_measure(mask_with_feature), sum_f=torch.sum)
                                impact_no_feature = compute_mean(impact_measure(mask_no_feature), sum_f=torch.sum)
                                delta_s_feature = map_collections(operator.sub, impact_with_feature, impact_no_feature)
                                delta_s_feature_zeros = map_collections(lambda val: torch.zeros_like(val), delta_s_feature)
                                delta_s_feature_pos = map_collections(max, delta_s_feature, delta_s_feature_zeros)
                                delta_s_feature_neg = map_collections(
                                    max,
                                    map_collections(operator.neg, delta_s_feature),
                                    delta_s_feature_zeros
                                )
                                if self.store_graph:
                                    self.graph[s_feature_edge] = (delta_s_feature_pos, delta_s_feature_neg)
                            else:
                                delta_s_feature_pos, delta_s_feature_neg = self.graph[s_feature_edge]

                            # computes (1/|S| \sum_{j in S} |x_j|^p)^{1 / p}
                            x_contrib_pos = map_collections(lambda val: theta_map(val, s_x, s_card, p), x_pos)
                            x_contrib_neg = map_collections(lambda val: theta_map(val, s_x, s_card, p), x_neg)
                            # expand x_contrib to a collection only if it is a tensor
                            # and the impact_measure is a multiple loss impact_measure
                            x_contrib_pos = expand_as(x_contrib_pos, delta_s_feature_pos)
                            x_contrib_neg = expand_as(x_contrib_neg, delta_s_feature_neg)

                            s_contrib_pos = map_collections(operator.mul, delta_s_feature_pos, x_contrib_pos)
                            s_contrib_neg = map_collections(operator.mul, delta_s_feature_neg, x_contrib_neg)
                            feature_pos_score = increment_score(feature_pos_score, s_contrib_pos)
                            feature_neg_score = increment_score(feature_neg_score, s_contrib_neg)

                    scores_pos[feature_idx] = feature_pos_score
                    scores_neg[feature_idx] = feature_neg_score

                return scores_pos, scores_neg

        def compute_eigenvector_centrality(x):
            feval = FEvaluator(self, store_graph=store_graph)
            norm_residual = 2 * residual_tol
            curr_iter = 1
            y_pos, y_neg = x, x
            while curr_iter <= niters and norm_residual > residual_tol:
                y_pos_curr, y_neg_curr = feval(y_pos, y_neg)
                y_pos_curr = (y_pos_curr * y_pos) ** (1 / (2 + 1e-5))
                y_neg_curr = (y_neg_curr * y_neg) ** (1 / (2 + 1e-5))

                y_pos_curr_norm = y_pos_curr.norm(p=p_normalization)
                if y_pos_curr_norm.item() > score_min_norm_threshold:
                    y_pos_curr = y_pos_curr / y_pos_curr_norm

                y_neg_curr_norm = y_neg_curr.norm(p=p_normalization)
                if y_neg_curr_norm.item() > score_min_norm_threshold:
                    y_neg_curr = y_neg_curr / y_neg_curr_norm

                # advance iteration
                norm_residual = max(
                    (y_pos - y_pos_curr).norm(p=p_normalization).item(),
                    (y_neg - y_neg_curr).norm(p=p_normalization).item()
                )
                curr_iter += 1
                y_pos, y_neg = y_pos_curr, y_neg_curr
            # scale by the eigenvalue
            return feval(y_pos, y_neg)

        init = torch.ones((len(feature_subset),), device=impact_measure.device_type) / len(feature_subset)
        if centrality == 'degree':
            scores = FEvaluator(self, store_graph=store_graph)(init, init)
        elif centrality == 'eigenvector':
            scores = compute_eigenvector_centrality(init)
        else:
            raise ValueError(f"{type(self)}.explain(): unsupported value for parameter `centrality`")
        score_pos, score_neg = scores
        return zip(score_pos - score_neg, score_pos, score_neg)


if __name__ == '__main__':
    from impact_measure import ImpactMeasure

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

    features_to_explain = [0, 2, 4, 6]
    hyperx = HyperXExplainer()
    for feature, explanation in zip(features_to_explain, hyperx.explain(im, feature_subset=features_to_explain)):
        print(f"feature: {feature}, explanation: {explanation}")
