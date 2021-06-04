import numpy as np
import onnx
import onnxruntime as rt
import torch
from hummingbird.ml import TorchContainer


def model_factory(classname, *args, **kwargs):
    try:
        cls = globals()[classname]
    except KeyError:
        return None
    return cls(*args, **kwargs)


class ModelInterface:
    def __init__(self):
        pass

    def load(self, path):
        pass

    def __call__(self, x, *args, **kwargs):
        # x can be a tensor or a numpy array
        # output is a tensor
        pass


class XGBoostRegression(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model = None
        self._ort_session = None
        self.out_type = torch.Tensor

    def load(self, path):
        onx = onnx.load(path)
        self._ort_session = rt.InferenceSession(onx.SerializeToString())

    def __call__(self, x, *args, **kwargs):
        x_np = None
        if type(x) == np.ndarray:
            x_np = x
        elif type(x) == torch.Tensor:
            x_np = x.numpy()
        if x_np is None:
            raise ValueError(F"Unknown input type '{type(x)}'")

        preds = self._ort_session.run(None, {'input': x_np.astype(np.float32)})[0]
        if self.out_type == torch.Tensor:
            return torch.tensor(preds)
        else:
            return preds


class TorchBlackBox(ModelInterface):
    def __init__(self):
        super().__init__()
        self._model = None
        self.out_type = torch.Tensor

    def load(self, path):
        self._model = torch.load(path)
        self._model.eval()

    def __call__(self, x, *args, **kwargs):
        if type(x) == np.ndarray:
            x_tensor = torch.tensor(x)
        elif type(x) == torch.Tensor:
            x_tensor = x
        else:
            raise ValueError(F"Unknown input type: '{type(x)}'")

        preds = self._model(x_tensor, *args, **kwargs)
        if self.out_type == torch.Tensor and type(preds) == np.ndarray:
            return torch.tensor(preds)
        elif self.out_type == np.ndarray and type(preds) == torch.Tensor:
            return preds.detach().numpy()
        return preds


class TorchHumXGBModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self._model = None
        self.out_type = torch.Tensor

    def load(self, path):
        self._model = TorchContainer.load(str(path))

    def __call__(self, x, *args, **kwargs):
        preds = self._model.predict(x)
        if self.out_type == torch.Tensor and type(preds) == np.ndarray:
            return torch.tensor(preds).reshape(preds.shape[0], 1)
        elif self.out_type == np.ndarray and type(preds) == torch.Tensor:
            return preds.numpy()
        return preds
