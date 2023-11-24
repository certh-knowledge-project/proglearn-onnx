import os

# Disable all tensorflow debugging logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import unittest
import sys
import os.path as osp
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from numpy.testing import assert_almost_equal
import packaging.version as pv

from onnx.checker import ValidationError
from onnxruntime import InferenceSession
from onnxruntime import __version__ as ort_version

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
except ImportError:
    # onnxruntime <= 0.5
    InvalidArgument = RuntimeError

from proglearn.transformers import TreeClassificationTransformer
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.voters import TreeClassificationVoter
from proglearn.progressive_learner import ClassificationProgressiveLearner
from proglearn.forest import LifelongClassificationForest

from prog2onnx import Prog2ONNX
from prog2onnx.utils._constants import (
    MIN_TARGET_OPSET,
    TARGET_AI_ONNX_ML,
    MAX_TARGET_OPSET,
)

# set the seed
np.random.seed(42)

# set the constants
TARGET_OPSET = MIN_TARGET_OPSET  # minimum working opset version
TARGET_OPSET_ML = TARGET_AI_ONNX_ML


def create_data(n_samples, n_features, n_classes):
    """
    Generate a synthetic dataset and split it into training and testing sets.

    Parameters
    ----------
    n_samples : int
        The total number of samples in the synthetic dataset.
    n_features : int
        The total number of features in the synthetic dataset.
    n_classes : int
        The number of class labels in the synthetic dataset.

    Returns
    -------
    X_train : np.ndarray
        The training data, an array of shape (n_samples * 0.7, n_features).
    X_test : np.ndarray
        The testing data, an array of shape (n_samples * 0.3, n_features).
    y_train : np.ndarray
        The training labels, an array of shape (n_samples * 0.7,).
    y_test : np.ndarray
        The testing labels, an array of shape (n_samples * 0.3,).

    Notes
    -----
    The function uses `sklearn.datasets.make_classification` to generate a synthetic dataset and
    `sklearn.model_selection.train_test_split` to split the dataset into training and testing sets.
    """
    n_samples = n_samples
    n_features = n_features
    n_classes = n_classes

    # Generate the synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_redundant=0,
        n_informative=n_features,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    return X_train, X_test, y_train, y_test


def create_model(X_train, y_train, tasks, num_transfs, task_id=0, kappa=None, **kwargs):
    """
    Create a ClassificationProgressiveLearner model and add tasks to it.

    Parameters
    ----------
    X_train : np.ndarray
        The training data.
    y_train : np.ndarray
        The training labels.
    tasks : int
        The number of tasks to add to the model.
    num_transfs : int
        The number of transformers to add for each task.
    task_id : int, optional
        The id of the task. Default is 0.
    kappa : float, optional
        The kappa value for the TreeClassificationVoter. If not provided, no kappa value is used.
    **kwargs : dict, optional
        Additional keyword arguments for the TreeClassificationTransformer.

    Returns
    -------
    model : ClassificationProgressiveLearner
        The created model with tasks added.

    Raises
    ------
    ValueError
        If the provided task_id is not in the model's task_ids.

    Notes
    -----
    The function uses `progressive_learner.ClassificationProgressiveLearner` to create a model and
    `progressive_learner.ClassificationProgressiveLearner.add_task` to add tasks to the model.
    """
    defaults_kwargs = {"random_state": 42}
    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs": defaults_kwargs if not kwargs else kwargs}
    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = dict() if not kappa else {"kappa": kappa}
    default_decider_class = SimpleArgmaxAverage

    model = ClassificationProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs={"classes": np.unique(y_train)},
    )

    for i in range(tasks):
        model.add_task(
            X=X_train,
            y=y_train,
            num_transformers=num_transfs,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={"classes": np.unique(y_train)},
        )

    if task_id not in model.get_task_ids():
        raise ValueError("Invalid task_id: %d" % task_id)

    model.task_id = task_id
    return model


def create_model_forest(X_train, y_train, tasks, estimators=50, task_id=0, **kwargs):
    # create model
    model = LifelongClassificationForest(default_n_estimators=estimators)
    for i in range(tasks):
        model.add_task(X_train, y_train, task_id=i)

    if task_id not in model.get_task_ids():
        raise ValueError("Invalid task_id: %d" % task_id)

    model.task_id = task_id
    return model


class TestProgressiveLearnerClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TEST_ONNX_FILE = "onnx_test.onnx"
        sys.setrecursionlimit(10000)

    @unittest.skipIf(
        pv.Version(ort_version.split("+")[0]) < pv.Version("1.8.0"),
        f"ONNX runtime >= 1.8.0 is required for TARGET_OPSET >= {TARGET_OPSET}",
    )
    def setUp(self):
        return super().setUp()

    def tearDown(self):
        if os.path.exists(osp.join(os.getcwd(), self.TEST_ONNX_FILE)):
            os.remove(osp.join(os.getcwd(), self.TEST_ONNX_FILE))
        else:
            pass

    def test_progLearn_binary_dt(self):
        KWARGS = {
            "criterion": "entropy",
            "splitter": "random",
            "max_depth": 16,
            "max_features": "log2",
            "random_state": 42,
            "class_weight": "balanced",
        }
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(
            X_train=X_train, y_train=y_train, tasks=2, num_transfs=2, **KWARGS
        )

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = Prog2ONNX(model).to_onnx(model.task_id)

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_binary(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(X_train=X_train, y_train=y_train, tasks=2, num_transfs=2)

        model_onnx = Prog2ONNX(model).to_onnx(model.task_id)
        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_lifelong(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=300, n_features=3, n_classes=2
        )

        # create model
        model = create_model_forest(X_train, y_train, 5)

        model_onnx = Prog2ONNX(model).to_onnx(model.task_id)
        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1], decimal=6)

    def test_progLearn_validate(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(X_train=X_train, y_train=y_train, tasks=2, num_transfs=2)
        p2o = Prog2ONNX(model)
        _ = p2o.to_onnx(model.task_id)

        try:
            p2o.validate()
        except ValidationError:
            self.fail("Invalid ONNX model")

    def test_progLearn_binary_opset(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(X_train=X_train, y_train=y_train, tasks=2, num_transfs=2)

        for opset in (MIN_TARGET_OPSET - 1, MAX_TARGET_OPSET + 1):
            with self.subTest(f"Error when running for opset={opset}", i=opset):
                with self.assertRaises(ValueError):
                    _ = Prog2ONNX(model).to_onnx(model.task_id, target_opset=opset)

    def test_progLearn_binary_save(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(X_train=X_train, y_train=y_train, tasks=2, num_transfs=2)

        # save to file
        p2o = Prog2ONNX(model)
        _ = p2o.to_onnx(model.task_id)
        p2o.save(self.TEST_ONNX_FILE)
        onnx_file = osp.join(os.getcwd(), self.TEST_ONNX_FILE)
        self.assertTrue(osp.exists(onnx_file))

        try:
            sess = InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n") from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_binary_kappa(self):
        KAPPA = 0.5
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(
            X_train=X_train, y_train=y_train, tasks=2, num_transfs=2, kappa=KAPPA
        )

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = Prog2ONNX(model).to_onnx(model.task_id)

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_binary_single_taskId(self):
        TASK_ID = 1

        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(
            X_train=X_train, y_train=y_train, tasks=2, num_transfs=2, task_id=TASK_ID
        )

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = Prog2ONNX(model).to_onnx(TASK_ID)

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=TASK_ID)
        expected_proba = model.predict_proba(X_test, task_id=TASK_ID)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_iris(self):
        X, y = load_iris(return_X_y=True)
        X = X.astype(np.float32)

        model = create_model(X_train=X, y_train=y, tasks=4, num_transfs=4)

        model_onnx = Prog2ONNX(model).to_onnx(model.task_id)

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        res = sess.run(None, {"float_input": X})
        self.assertEqual(
            model.predict(X, task_id=model.task_id).tolist(), res[0].tolist()
        )
        assert_almost_equal(model.predict_proba(X, task_id=model.task_id), res[1])

    def test_progLearn_binary_all_taskIds(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model w/ 5 task ids
        model = create_model(X_train=X_train, y_train=y_train, tasks=5, num_transfs=3)
        n_features = model.task_id_to_X[0].shape[-1]

        for i in model.get_task_ids():
            with self.subTest(f"Error when running for task_id={i}", i=i):
                model.task_id = i  # set the task_id
                model_onnx = Prog2ONNX(model).to_onnx(model.task_id)

                self.assertTrue(model_onnx is not None)
                try:
                    sess = InferenceSession(
                        model_onnx.SerializeToString(),
                        providers=["CPUExecutionProvider"],
                    )
                except InvalidArgument as e:
                    raise AssertionError(
                        "Cannot load model\n%r" % str(model_onnx)
                    ) from e

                res = sess.run(None, {"float_input": X_test})
                self.assertEqual(
                    model.predict(X_test, task_id=i).tolist(), res[0].tolist()
                )
                assert_almost_equal(model.predict_proba(X_test, task_id=i), res[1])

    def test_progLearn_multi(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=300, n_features=5, n_classes=4
        )

        # create model
        model = create_model(X_train=X_train, y_train=y_train, tasks=4, num_transfs=3)

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = Prog2ONNX(model).to_onnx(model.task_id)

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])


if __name__ == "__main__":
    unittest.main(verbosity=2, catchbreak=True)
