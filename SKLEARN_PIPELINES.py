# -*- coding: utf-8 -*-
"""
SKLearn NIR pipelines for SenSoRTC.

Supports EVK SQALAR .mat files and SenSoRTC .npy raw spectral chunks.
Canonical spectral cube layout for both formats:
    (width, bands, lines)
Example:
    (312, 220, n_lines)

Training format after loading:
    X = (n_spectra, bands)
    y = (n_spectra,)

Design:
    NIRPipelineBase contains all reusable loading/training/evaluation/saving logic.
    Concrete subclasses only define build_pipeline().
"""

from pathlib import Path

import joblib
import numpy as np
from scipy.io import loadmat

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from scipy_transformer_adapter import SavGolTransformer


class NIRPipelineBase:
    """Base class for SenSoRTC/EVK NIR sklearn classifiers."""

    def __init__(
        self,
        window_length=15,
        polyorder=2,
        deriv=1,
        random_state=42,
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.random_state = random_state

        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.class_names = None

        self.pipe = self.build_pipeline()

    def build_pipeline(self):
        """Subclasses must return an sklearn Pipeline."""
        raise NotImplementedError

    def preprocessing_steps(self):
        """Reusable preprocessing for all NIR spectral classifiers."""
        return [
            ("savgol", SavGolTransformer(
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv,
            )),
            ("zscore", StandardScaler()),
        ]

    @staticmethod
    def _material_name_from_path(path):
        stem = Path(path).stem
        if stem.endswith("_timestamps"):
            raise ValueError(f"Do not pass timestamp files into training: {path}")
        return stem

    @staticmethod
    def _array_to_spectra_legacy_or_evk(arr, source_name="array"):
        """Convert spectral arrays to X=(n_spectra, bands).

        EVK SQALAR / new SenSoRTC:
            (width, bands, lines), e.g. (312, 220, nLines)

        Old SenSoRTC before transpose change:
            (lines, width, bands), e.g. (nLines, 312, 220)

        2-D matrices are assumed to already be:
            (n_spectra, bands)
        """
        arr = np.asarray(arr)

        if arr.ndim == 3:
            shape = arr.shape

            # EVK/new SenSoRTC: (width, bands, lines)
            if shape[1] == 220:
                bands = shape[1]
                return arr.transpose(0, 2, 1).reshape(-1, bands)

            # Old SenSoRTC: (lines, width, bands)
            if shape[2] == 220:
                bands = shape[2]
                return arr.reshape(-1, bands)

            # Fallback: choose a plausible spectral axis. Prefer middle axis.
            candidate_axes = [i for i, s in enumerate(shape) if 16 <= s <= 4096]
            if not candidate_axes:
                raise ValueError(f"Cannot infer spectral axis for {source_name}: {shape}")

            spectral_axis = 1 if 1 in candidate_axes else candidate_axes[-1]
            bands = shape[spectral_axis]
            moved = np.moveaxis(arr, spectral_axis, -1)
            return moved.reshape(-1, bands)

        if arr.ndim == 2:
            return arr

        raise ValueError(f"Unsupported ndim for {source_name}: {arr.ndim}, shape={arr.shape}")

    def _filter_background(self, X, min_mean):
        X = np.asarray(X, dtype=np.float32)
        keep = X.mean(axis=1) >= float(min_mean)
        X = X[keep]
        if X.size == 0:
            raise ValueError(
                f"No spectra left after min_mean={min_mean}. "
                "Lower min_mean or check whether this file contains foreground material."
            )
        return X

    def load_imnData_mat(self, path, label, min_mean=800):
        path = Path(path)
        mat = loadmat(path)

        if "imnData" in mat:
            arr = mat["imnData"]
        else:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            if not keys:
                raise ValueError(f"No data arrays found in MAT file: {path}")
            arr = mat[keys[0]]

        X = self._array_to_spectra_legacy_or_evk(arr, source_name=str(path))
        X = self._filter_background(X, min_mean=min_mean)
        y = np.full(X.shape[0], label, dtype=np.uint8)
        return X, y

    def load_npy_spectral(self, path, label, min_mean=800):
        path = Path(path)
        if path.stem.endswith("_timestamps"):
            raise ValueError(f"This is a timestamp file, not spectral data: {path}")

        arr = np.load(path)
        X = self._array_to_spectra_legacy_or_evk(arr, source_name=str(path))
        X = self._filter_background(X, min_mean=min_mean)
        y = np.full(X.shape[0], label, dtype=np.uint8)
        return X, y

    def load_spectral_file(self, path, label, min_mean=800):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".mat":
            return self.load_imnData_mat(path, label, min_mean=min_mean)

        if suffix == ".npy":
            return self.load_npy_spectral(path, label, min_mean=min_mean)

        raise ValueError(f"Unsupported file type: {path}")

    def load_data(self, paths, min_mean=800, test_size=0.2, random_state=None):
        if random_state is None:
            random_state = self.random_state

        X_parts = []
        y_parts = []

        for class_idx, path in enumerate(paths, start=1):
            X_data, y_data = self.load_spectral_file(
                path,
                class_idx,
                min_mean=min_mean,
            )
            print(f"Loaded {Path(path).name}: X={X_data.shape}, label={class_idx}")
            X_parts.append(X_data)
            y_parts.append(y_data)

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

    def train(self, paths, name=None, min_mean=800, test_size=0.2):
        if name is None:
            name = f"{self.__class__.__name__}.joblib"

        self.load_data(paths, min_mean=min_mean, test_size=test_size)
        self.pipe.fit(self.X_train, self.y_train)

        pred = self.pipe.predict(self.X_test)
        print(confusion_matrix(self.y_test, pred))

        material_names = [self._material_name_from_path(p) for p in paths]
        print(classification_report(
            self.y_test,
            pred,
            labels=list(range(1, len(paths) + 1)),
            target_names=material_names,
        ))

        bundle = {
            "pipeline": self.pipe,
            "class_names": ["Background"] + material_names,
            "class_labels": list(range(len(paths) + 1)),
            "kind": self.__class__.__name__,
            "format": "sklearn_pipeline_bundle_v1",
            "preprocessing": {
                "savgol_window_length": self.window_length,
                "savgol_polyorder": self.polyorder,
                "savgol_deriv": self.deriv,
                "zscore": "StandardScaler",
            },
        }

        joblib.dump(bundle, name)
        print(f"Saved: {name}")
        return self.pipe


class Shallow_NN(NIRPipelineBase):
    def __init__(
        self,
        window_length=15,
        polyorder=2,
        deriv=1,
        hidden_layer_sizes=(64, 32),
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        super().__init__(
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            random_state=random_state,
        )

    def build_pipeline(self):
        return Pipeline(
            self.preprocessing_steps() + [
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    random_state=self.random_state,
                )),
            ]
        )


class SVM_RBF(NIRPipelineBase):
    def __init__(
        self,
        window_length=15,
        polyorder=2,
        deriv=1,
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    ):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        super().__init__(
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            random_state=random_state,
        )

    def build_pipeline(self):
        return Pipeline(
            self.preprocessing_steps() + [
                ("svm", SVC(
                    kernel="rbf",
                    C=self.C,
                    gamma=self.gamma,
                    class_weight=self.class_weight,
                )),
            ]
        )


class SVM_Linear(NIRPipelineBase):
    def __init__(
        self,
        window_length=15,
        polyorder=2,
        deriv=1,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    ):
        self.C = C
        self.class_weight = class_weight
        super().__init__(
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            random_state=random_state,
        )

    def build_pipeline(self):
        return Pipeline(
            self.preprocessing_steps() + [
                ("linear_svm", LinearSVC(
                    C=self.C,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    max_iter=10000,
                )),
            ]
        )
