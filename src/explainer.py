import numpy as np
import shap
import lime
import torch
import hyperx.pytorch as pp


class Explainer:
    def __init__(self):
        pass

    def explain(self, explain_params, instance, baseline, model):
        pass


class HyperXExplainer(Explainer):
    def __init__(self):
        super().__init__()
        self.explainer = pp.HyperXExplainer()

    def explain(self, explain_params, instance, baseline, model):
        instance_torch = [(torch.tensor(instance), torch.zeros(1, 1))]
        baseline_torch = [(torch.tensor(baseline), torch.zeros(*baseline.shape))]
        impact_measure = pp.ImpactMeasure.build(
            instance_torch, baseline_torch, model, loss=None
        )
        model.out_type = torch.Tensor
        explanation_generator = self.explainer.explain(impact_measure, [f for f in range(instance.shape[1])], **explain_params)
        return explanation_generator


class ShapleyPkgExplainer(Explainer):
    def __init__(self):
        super().__init__()

    def explain(self, explain_params, instance, baseline, model):

        model.out_type = np.ndarray
        exp_model = model

        np.random.seed(explain_params["random_state"])
        if explain_params["type"] == "kernel_shap":
            explainer = shap.KernelExplainer(exp_model, baseline)
        elif explain_params["type"] == "sampling_shap":
            explainer = shap.SamplingExplainer(exp_model, baseline)
        elif explain_params["type"] == "perm_shap":
            explainer = shap.PermutationExplainer(exp_model, baseline)
        elif explain_params["type"] == "random_shap":
            explainer = shap.other.Random()
        else:
            raise ValueError(F"Unknown explainer: {explain_params['type']}.")

        if explain_params["type"] not in ["perm_shap", "random_shap"]:
            shap_values = explainer.explain(instance, nsamples=explain_params['nsamples'])
        elif explain_params["type"] == "random_shap":
            shap_values = explainer.attributions(instance).reshape(-1, 1)
        else:
            try:
                # the call to .shap_values throws an exception, therefore using .explain_row instead
                # here we call .explain_row with the same arguments that are used when calling .shap_values
                # shap_values = explainer.shap_values(instance, npermutations=explain_params['nsamples'])
                shap_values = explainer.explain_row(instance.reshape(-1),
                                                    max_evals=explain_params['nsamples'] * instance.shape[1],
                                                    main_effects=[],
                                                    error_bounds=False,
                                                    batch_size=10, outputs=None, silent=False)
                shap_values = shap_values['values'].reshape(-1, 1)
            except Exception as e:
                print(e)
                shap_values = np.empty((instance.shape[1],))
                shap_values.fill(np.nan)

        return shap_values


class LimePkgExplainer(Explainer):
    def __init__(self):
        super().__init__()

    def explain(self, explain_params, instance, baseline, model):

        model.out_type = np.ndarray
        explainer = lime.lime_tabular.LimeTabularExplainer(
                        baseline,
                        categorical_features=explain_params['categorical_features'],
                        mode="regression",
                        random_state=explain_params['random_state'])

        exp_model = model
        num_features = instance.shape[1]
        lime_exp = explainer.explain_instance(instance.reshape(-1),
                                              exp_model,
                                              num_features=num_features,
                                              num_samples=explain_params['num_samples'])

        lime_list_exp = lime_exp.local_exp[lime_exp.dummy_label]
        scores = [None] * num_features
        for item in lime_list_exp:
            scores[item[0]] = item[1]
        explanation_generator = scores
        return explanation_generator
