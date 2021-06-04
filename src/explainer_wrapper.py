"""
    Estimation of Lipschitz metric was adapted from: https://github.com/dmelis/robust_interpret
"""

from functools import partial

import lime
import numpy as np
import shap
import torch
from lime import lime_tabular
from skopt import gp_minimize

from hyperx.pytorch import HyperXExplainer, ImpactMeasure


def _parallel_lipschitz(wrapper, i, x, bound_type, eps, n_calls):
    print('\n\n ***** PARALLEL : Example ' + str(i) + '********')
    print(wrapper.net.__dict__.keys())
    l, _ = wrapper.local_lipschitz_estimate(x, eps=eps, bound_type=bound_type, n_calls=n_calls)
    return l


class ExplainerWrapper():
    def __init__(self, model, model_type="classification", multiclass=False,
                 feature_names=None, class_names=None, baseline=None,
                 background_data=None, seed=42):
        self.model_type = model_type
        self.model = model
        self.feature_names = feature_names
        self.multiclass = multiclass
        self.class_names = class_names
        self.baseline = baseline
        self.background_data = background_data
        self.seed = seed
        if self.background_data is not None:
            print("Computing train data stats...")
            self.train_stats = {
                'min': self.background_data[0].min(0),
                'max': self.background_data[0].max(0),
                'mean': self.background_data[0].mean(0),
                'std': self.background_data[0].std(0)
            }

    def lipschitz_ratio(self, fx, x1, x2, reshape=None, minus=False):
        """
            If minus = True, returns minus this quantitiy.

            || f(x) - f(y) ||/||x - y||

        """
        # NEed this ungly hack because skopt sends lists
        if type(x1) is list:
            x1 = np.array(x1)
        if type(x2) is list:
            x2 = np.array(x2)
        if reshape is not None:
            # Necessary because gpopt requires to flatten things, need to restrore expected sshape here
            x1 = x1.reshape(reshape)
            x2 = x2.reshape(reshape)
        # print(x.shape, x.ndim)
        multip = -1 if minus else 1
        return multip * np.linalg.norm(fx - self.explain(x2)) / np.linalg.norm(x1 - x2)

    def local_lipschitz_estimate(self, x, y=None, optim='gp', eps=None, bound_type='box',
                                 clip=True, n_calls=100, njobs=-1, verbose=False):
        """
            Compute one-sided lipschitz estimate for explainer. Adequate for local
             Lipschitz, for global must have the two sided version. This computes:

                max_z || f(x) - f(z)|| / || x - z||

            Instead of:

                max_z1,z2 || f(z1) - f(z2)|| / || z1 - z2||

            If eps provided, does local lipzshitz in:
                - box of width 2*eps along each dimension if bound_type = 'box'
                - box of width 2*eps*va, along each dimension if bound_type = 'box_norm' (i.e. normalize so that deviation is eps % in each dim )
                - box of width 2*eps*std along each dimension if bound_type = 'box_std'

            max_z || f(x) - f(z)|| / || x - z||   , with f = theta

            clip: clip bounds to within (min, max) of dataset

        """
        # Compute bounds for optimization
        if eps is None:
            # If want to find global lipzhitz ratio maximizer - search over "all space" - use max min bounds of dataset fold of interest
            # gp can't have lower bound equal upper bound - so move them slightly appart
            lwr = self.train_stats['min'].flatten() - 1e-6
            upr = self.train_stats['max'].flatten() + 1e-6
        elif bound_type == 'box':
            lwr = (x - eps).flatten()
            upr = (x + eps).flatten()
        else:
            # gp can't have lower bound equal upper bound - so set min std to 0.001
            lwr = (
                    x - eps * np.maximum(self.train_stats['std'], 0.001)).flatten()
            upr = (
                    x + eps * np.maximum(self.train_stats['std'], 0.001)).flatten()
        if clip:
            lwr = lwr.clip(min=self.train_stats['min'].min())
            upr = upr.clip(max=self.train_stats['max'].max())
        bounds = list(zip(*[lwr, upr]))
        if x.ndim > 2:
            # This is an image, will need to reshape
            orig_shape = x.shape
            x = x.flatten()
        else:
            orig_shape = x.shape

        f_x = self.explain(x, y)

        # Run optimization
        print('Running BlackBox Minimization with Bayesian Optimization')
        # Need minus because gp only has minimize method
        f = partial(self.lipschitz_ratio, f_x, x, reshape=orig_shape, minus=True)
        res = gp_minimize(f, bounds, n_calls=n_calls,
                          verbose=verbose, n_jobs=njobs, random_state=self.seed)

        lip, x_opt = -res['fun'], np.array(res['x'])
        if verbose:
            print(lip, np.linalg.norm(x - x_opt))
        return lip

    def explain(self, x, y=None, *args, **kwargs):
        return np.array([])


class HyperxExplainerWrapper(ExplainerWrapper):
    def __init__(self, model, model_type="classification", multiclass=False,
                 feature_names=None, class_names=None, baseline=None,
                 background_data=None, seed=42,
                 explainer_params=None):
        super().__init__(model, model_type, multiclass, feature_names, class_names, baseline, background_data, seed)

        self.explainer = None
        self.params = explainer_params

    def explain(self, x, y=None, *args, **kwargs):
        self.explainer = HyperXExplainer()
        if y is None:
            y = self.model(x)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        behaviors = [(x_tensor, y_tensor)]

        baselines = [(torch.tensor(self.baseline[0]), torch.tensor(self.baseline[1]))]
        metric = ImpactMeasure.build(
            behaviors, baselines, self.model, loss=None
        )
        explanation_generator = self.explainer.explain(metric, **self.params)
        attributes = np.zeros(x_tensor.shape[1])
        for feature_idx, scores in enumerate(explanation_generator):
            (score, score_pos, score_neg) = scores
            attributes[feature_idx] = score.numpy()
        return attributes


class ShapExplainerWrapper(ExplainerWrapper):
    """
        Wrapper around SHAP explanation framework from shap github package by the authors
    """

    def __init__(self, model, model_type, multiclass=False, feature_names=None,
                 class_names=None, baseline=None, background_data=None, seed=42, nsamples=100):
        print('Initializing SHAP explainer wrapper')
        super().__init__(model, model_type, multiclass, feature_names, class_names, baseline, background_data,
                         seed=seed)

        # x_train_summary = shap.kmeans(background_data, 10)
        self.explainer = shap.KernelExplainer(model, self.baseline[0])
        self.nsamples = nsamples

    def explain(self, x, y=None, *args, **kwargs):
        """
            y only needs to be specified in the multiclass case. In that case,
            it's the class to be explained (typically, one would take y to be
            either the predicted class or the true class). If it's a single value,
            same class explained for all inputs
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y is None:
            # Explain predicted class
            if self.class_names is None:
                shape_len = 1
            else:
                shape_len = len(self.class_names)
            y = np.argmax(self.model(x).reshape(x.shape[0], shape_len), axis=1)
        elif y is not None and y.ndim > 1:
            # Means y is given in one-hot format
            y = np.argwhere(y)[:, 1]
        elif (type(y) is int or y.ndim == 0) and (x.ndim == 1):
            # Single example
            # y = np.array([y]).reshape(1,1)
            y = [y]
            x = x.reshape(1, -1)
        elif type(y) is int or y.ndim == 0:
            # multiple examples, same class to be explained in all of them
            y = [y] * x.shape[0]

        assert x.shape[0] == len(
            y), "x.shape = {}, len(y) = {}".format(x.shape, len(y))

        exp_classes = self.explainer.shap_values(x, nsamples=self.nsamples, verbose=False)

        if self.multiclass:
            exp = np.array([exp_classes[y[i]][i]
                            for i in range(len(exp_classes[0]))])
        else:
            exp = exp_classes

        if x.shape[0] == 1:
            # Squeeze if single prediction
            exp = exp[0]

        exp_dict = dict(zip(self.feature_names + ['bias'], exp.T.tolist()))
        vals = np.array([exp_dict[feat]
                         for feat in self.feature_names if feat in exp_dict.keys()]).T
        return vals


class LimeExplainerWrapper(ExplainerWrapper):
    """
        Wrapper around LIME explanation framework from lime github package by the authors
    """

    def __init__(self, model, model_type="classification", multiclass=False, feature_names=None,
                 class_names=None, baseline=None, background_data=None, seed=42, num_samples=100,
                 feature_selection='auto',
                 num_features=None,
                 categorical_features=None, verbose=False):
        print('Initializing LIME explainer wrapper')
        super().__init__(model, model_type, multiclass,
                         feature_names, class_names, baseline, background_data, seed)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            self.baseline[0], feature_names=feature_names, class_names=class_names,
            discretize_continuous=False, categorical_features=categorical_features,
            verbose=verbose, mode=model_type, feature_selection=feature_selection)

        self.num_features = num_features if num_features else len(self.feature_names)
        self.num_samples = num_samples

    def extract_att_tabular(self, exp, y):
        # """ Method for extracting a numpy array from lime explanation objects"""
        # if x.ndim == 1:
        # Single explanation
        if y is None or (type(y) is list and y[0] is None):
            # exp_dict = dict(exp.as_list()) # SOMETIMES BREAKS
            if self.model_type == "regression":
                exp_dict = dict(exp.as_list())
            else:
                exp_dict = dict(exp.as_list(exp.top_labels[0]))  # Hacky but works
        # Hacky but works
        elif self.multiclass:
            exp_dict = dict(exp.as_list(label=y))
        else:
            exp_dict = dict(exp.as_list())
        # exp_dict = dict(exp.as_list())
        # FOR NOW IGNORE DISCRETE - THEY're
        vals = np.array([exp_dict[feat]
                         for feat in self.feature_names if feat in exp_dict.keys()])
        return vals

    def explain(self, x, y=None, x_raw=None, return_dict=False, show_plot=False, *args, **kwargs):

        # the only thing that will change is labels and toplabels

        # if y is None:
        #     # No target class provided - use predicted
        #     y = self.model(x).argmax(1)
        #
        if x.ndim == 1:
            if not self.multiclass:
                labs, top_labs = [(1,)], None  # Explain the "only" class
            elif y is None:
                labs, top_labs = None, 1  # Explain only most likely predicted class
            else:
                labs, top_labs = (y,), None  # Explains class y
            exp = self.explainer.explain_instance(x, self.model,
                                                  labels=labs,
                                                  top_labels=top_labs,
                                                  num_features=self.num_features,
                                                  num_samples=self.num_samples,
                                                  )
            attributions = self.extract_att_tabular(exp, y)
            # self.explanation = exp
        else:
            # There's multiple examples to explain
            N = int(x.shape[0])
            if not self.multiclass:
                labs = [(1,)] * N
                top_labs = 1
            elif y is None:
                # Irrelevant, will be ignored with top_labs provided
                labs = [(None,)] * N
                top_labs = 1
            else:
                top_labs = None
                labs = [(y[i],) for i in range(N)]

            exp = [
                self.explainer.explain_instance(x[i, :],
                                                self.model,
                                                labels=labs[i], top_labels=top_labs,
                                                num_features=self.num_features,
                                                num_samples=self.num_samples,
                                                )
                for i in range(N)]

            attributions = [self.extract_att_tabular(
                exp[i], labs[i][0]) for i in range(len(exp))]

            # vals = np.stack(vals, axis = 1)
            attributions = np.stack(attributions, axis=0)

        # pdb.set_trace()
        if attributions.ndim > 1:
            # WHY DO WE NEED THIS? SEEMSS LIKE ELSE version is the way to unifoirmize one vs multiple examples
            # Was this needed for images?
            attributions = attributions.reshape(attributions.shape[0], -1)
        # else:
        #     attributions = attributions.reshape(1, attributions.shape[0])

        return attributions
