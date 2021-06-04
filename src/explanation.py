import pandas as pd
import torch
import os


# Interface to write to and read from a file explanations of various explainers
class ExplanationInterface:
    def __init__(self, explanation_gen, explain_params, feature_name_dict):
        pass

    def save(self, dir_path):
        pass

    def load(self, file_path):
        # Load explanation stored at <file_path>
        pass

    @property
    def importance_scores(self):
        # Return importance scores as a torch tensor
        pass

    @property
    def feature_names(self):
        names = [''] * self.explanation_df.shape[0]
        for idx, row in self.explanation_df.iterrows():
            names[int(row["Feature_Idx"])] = row["Feature_Name"]
        return names


class ShapleyExplanation(ExplanationInterface):
    def __init__(self, explanation_gen=None, explain_params=None, feature_name_dict=None):
        super().__init__(explanation_gen, explain_params, feature_name_dict)
        self.explanation_gen = explanation_gen
        self.explain_params = explain_params
        self.feature_name_dict = feature_name_dict  # dictionary <feature_idx>: feature_name
        self.explanation_df = None

    # TODO: does not work with multiloss metric
    def save(self, path):
        # Explanation for each feature looks like this: (<singleton tensor>, <singleton tensor>)
        # TODO: multiloss
        # TODO: save into files with names=explainer+explain params
        explanation_df = None
        for feature_idx, explanation in enumerate(self.explanation_gen):
            if explanation_df is None:
                columns = ["Feature_Idx", "Feature_Name", "Value", "Value_Abs"]
                explanation_df = pd.DataFrame(columns=columns)

            feature_all_idx = self.explain_params['feature_subset'][feature_idx]
            explanation_df = explanation_df.append({"Feature_Idx": feature_all_idx,
                                                    "Feature_Name": self.feature_name_dict[feature_all_idx],
                                                    "Value": explanation[0][0].item(),
                                                    "Value_Abs": explanation[1][0].item()}, ignore_index=True)

        if not os.path.exists(path.parents[0]):
            os.makedirs(path.parents[0])
        explanation_df.to_csv(path, index=False)

    def load(self, file_path):
        self.explanation_df = pd.read_csv(file_path, header=0)

    @property
    def importance_scores(self):
        scores = torch.empty(self.explanation_df.shape[0])
        scores.fill_(float("-Inf"))
        for idx, row in self.explanation_df.iterrows():
            scores[int(row["Feature_Idx"])] = row["Value"]
        return scores


class FirstOrderExplanation(ExplanationInterface):
    def __init__(self, explanation_gen=None, explain_params=None, feature_name_dict=None):
        super().__init__(explanation_gen, explain_params, feature_name_dict)
        self.explanation_gen = explanation_gen
        self.explain_params = explain_params
        self.feature_name_dict = feature_name_dict  # dictionary <feature_idx>: feature_name
        self.explanation_df = None

    def save(self, path):
        columns = ["Feature_Idx", "Feature_Name", "Phi", "Varphi"]
        explanation_df = pd.DataFrame(columns=columns)
        for feature_idx, explanation in enumerate(self.explanation_gen):
            if explanation_df is None:
                explanation_df = pd.DataFrame(columns=columns)

            feature_all_idx = self.explain_params['feature_subset'][feature_idx]
            explanation_df = explanation_df.append({"Feature_Idx": feature_all_idx,
                                                    "Feature_Name": self.feature_name_dict[feature_all_idx],
                                                    "Phi": explanation[0].item(),
                                                    "Varphi": explanation[1].item()}, ignore_index=True)

        if not os.path.exists(path.parents[0]):
            os.makedirs(path.parents[0])
        explanation_df.to_csv(path, index=False)

    def load(self, file_path):
        self.explanation_df = pd.read_csv(file_path, header=0)
        pass

    @property
    def importance_scores(self):
        scores = torch.empty(self.explanation_df.shape[0])
        scores.fill_(float("-Inf"))
        for idx, row in self.explanation_df.iterrows():
            scores[int(row["Feature_Idx"])] = row["Varphi"]
        return scores


class LIMEExplanation(ExplanationInterface):
    def __init__(self, explanation_gen=None, explain_params=None, feature_name_dict=None):
        super().__init__(explanation_gen, explain_params, feature_name_dict)
        self.explanation_gen = explanation_gen
        self.explain_params = explain_params
        self.feature_name_dict = feature_name_dict  # dictionary <feature_idx>: feature_name
        self.explanation_df = None

    # u_hat, lambda_empty, lambda_empty_eff, lambda_full, lambda_full_approx, error
    def save(self, path):

        explanation_add_df = pd.Series(index=["Lambda_0", "Lambda_0_Eff", "Lambda_Full", "Lambda_Full_Approx", "Error"])
        columns = ["Feature_Idx", "Feature_Name", "Value"]
        explanation_df = pd.DataFrame(columns=columns)

        explanation = next(self.explanation_gen)
        explanation_add_df["Lambda_0"] = explanation[1]
        explanation_add_df["Lambda_0_Eff"] = explanation[2]
        explanation_add_df["Lambda_Full"] = explanation[3]
        explanation_add_df["Lambda_Full_Approx"] = explanation[4]
        explanation_add_df["Error"] = explanation[5]

        for idx in range(explanation[0].shape[0]):
            feature_idx = self.explain_params['feature_subset'][idx]
            explanation_df = explanation_df.append({"Feature_Idx": feature_idx,
                                                    "Feature_Name": self.feature_name_dict[feature_idx],
                                                    "Value": explanation[0][idx].item()}, ignore_index=True)

        if not os.path.exists(path.parents[0]):
            os.makedirs(path.parents[0])
        explanation_df.to_csv(path, index=False)
        add_path = path.parents[0] / F"{path.stem}_add.csv"
        explanation_add_df.to_csv(add_path)

    def load(self, file_path):
        self.explanation_df = pd.read_csv(file_path, header=0)
        pass

    @property
    def importance_scores(self):
        scores = torch.empty(self.explanation_df.shape[0])
        scores.fill_(float("-Inf"))
        for idx, row in self.explanation_df.iterrows():
            scores[int(row["Feature_Idx"])] = row["Value"]
        return scores


class HyperXExplanation(ExplanationInterface):
    def __init__(self, explanation_gen=None, explain_params=None, feature_name_dict=None):
        super().__init__(explanation_gen, explain_params, feature_name_dict)
        self.explanation_gen = explanation_gen
        self.explain_params = explain_params
        self.feature_name_dict = feature_name_dict  # dictionary <feature_idx>: feature_name
        self.explanation_df = None

    def save(self, path):
        columns = ["Feature_Idx", "Feature_Name", "Pos-Neg", "Pos", "Neg"]
        explanation_df = pd.DataFrame(columns=columns)
        explanation = self.explanation_gen

        for feature_idx, scores in enumerate(explanation):
            (score, score_pos, score_neg) = scores
            explanation_df = explanation_df.append({"Feature_Idx": feature_idx,
                                                    "Feature_Name": self.feature_name_dict[feature_idx],
                                                    "Pos-Neg": score.item(),
                                                    "Pos": score_pos.item(),
                                                    "Neg": score_neg.item()}, ignore_index=True)

        if not os.path.exists(path.parents[0]):
            os.makedirs(path.parents[0])
        explanation_df.to_csv(path, index=False)

    def load(self, file_path):
        self.explanation_df = pd.read_csv(file_path, header=0)
        pass

    @property
    def importance_scores(self):
        scores = torch.empty(self.explanation_df.shape[0])
        scores.fill_(float("-Inf"))
        for idx, row in self.explanation_df.iterrows():
            scores[int(row["Feature_Idx"])] = row["Pos-Neg"]
        return scores


class ShapleyPkgExplanation(ExplanationInterface):
    def __init__(self, explanation_gen=None, explain_params=None, feature_name_dict=None):
        super().__init__(explanation_gen, explain_params, feature_name_dict)
        self.explanation_gen = explanation_gen
        self.explain_params = explain_params
        self.feature_name_dict = feature_name_dict  # dictionary <feature_idx>: feature_name
        self.explanation_df = None

    def save(self, path):
        columns = ["Feature_Idx", "Feature_Name", "Value"]
        explanation_df = pd.DataFrame(columns=columns)
        explanation = self.explanation_gen

        for feature_idx, score in enumerate(explanation):
            explanation_df = explanation_df.append({"Feature_Idx": feature_idx,
                                                    "Feature_Name": self.feature_name_dict[feature_idx],
                                                    "Value": score.item()}, ignore_index=True)

        if not os.path.exists(path.parents[0]):
            os.makedirs(path.parents[0])
        explanation_df.to_csv(path, index=False)

    def load(self, file_path):
        self.explanation_df = pd.read_csv(file_path, header=0)
        pass

    @property
    def importance_scores(self):
        scores = torch.empty(self.explanation_df.shape[0])
        scores.fill_(float("-Inf"))
        for idx, row in self.explanation_df.iterrows():
            scores[int(row["Feature_Idx"])] = row["Value"]
        return scores


class LimePkgExplanation(ExplanationInterface):
    def __init__(self, explanation_gen=None, explain_params=None, feature_name_dict=None):
        super().__init__(explanation_gen, explain_params, feature_name_dict)
        self.explanation_gen = explanation_gen
        self.explain_params = explain_params
        self.feature_name_dict = feature_name_dict  # dictionary <feature_idx>: feature_name
        self.explanation_df = None

    def save(self, path):
        columns = ["Feature_Idx", "Feature_Name", "Value"]
        explanation_df = pd.DataFrame(columns=columns)
        explanation = self.explanation_gen

        for feature_idx, score in enumerate(explanation):
            explanation_df = explanation_df.append({"Feature_Idx": feature_idx,
                                                    "Feature_Name": self.feature_name_dict[feature_idx],
                                                    "Value": score}, ignore_index=True)

        if not os.path.exists(path.parents[0]):
            os.makedirs(path.parents[0])
        explanation_df.to_csv(path, index=False)

    def load(self, file_path):
        self.explanation_df = pd.read_csv(file_path, header=0)
        pass

    @property
    def importance_scores(self):
        scores = torch.empty(self.explanation_df.shape[0])
        scores.fill_(float("-Inf"))
        for idx, row in self.explanation_df.iterrows():
            scores[int(row["Feature_Idx"])] = row["Value"]
        return scores
