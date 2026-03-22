"""Tests for olmlx.engine.flash.predictor and calibrate."""

import mlx.core as mx

from olmlx.engine.flash.predictor import PredictorBank, SparsityPredictor


class TestSparsityPredictor:
    def test_output_shape(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        scores = pred(x)
        mx.eval(scores)
        assert scores.shape == (1, inter)

    def test_scores_are_between_0_and_1(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((4, hidden))
        scores = pred(x)
        mx.eval(scores)
        assert mx.all(scores >= 0).item()
        assert mx.all(scores <= 1).item()

    def test_predict_active_respects_min_neurons(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        min_n = 16
        indices = pred.predict_active(x, threshold=0.99, min_neurons=min_n)
        mx.eval(indices)
        assert indices.shape[0] >= min_n

    def test_predict_active_respects_max_neurons(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        max_n = 4
        indices = pred.predict_active(x, threshold=0.0, max_neurons=max_n)
        mx.eval(indices)
        assert indices.shape[0] <= max_n

    def test_predict_active_returns_sorted_indices(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        indices = pred.predict_active(x, threshold=0.5, min_neurons=4)
        mx.eval(indices)
        arr = indices.tolist()
        assert arr == sorted(arr)

    def test_predict_active_indices_in_valid_range(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        indices = pred.predict_active(x, threshold=0.3, min_neurons=8)
        mx.eval(indices)
        assert mx.all(indices >= 0).item()
        assert mx.all(indices < inter).item()


class TestPredictorBank:
    def test_save_load_roundtrip(self, tmp_path):
        hidden, inter, rank = 16, 32, 4
        num_layers = 3

        bank = PredictorBank(num_layers, hidden, inter, rank)
        save_path = tmp_path / "predictors"
        bank.save(save_path)

        loaded = PredictorBank.load(save_path)
        assert len(loaded.predictors) == num_layers

        # Verify weights match
        x = mx.random.normal((1, hidden))
        for i in range(num_layers):
            orig_scores = bank.predictors[i](x)
            loaded_scores = loaded.predictors[i](x)
            mx.eval(orig_scores, loaded_scores)
            assert mx.allclose(orig_scores, loaded_scores, atol=1e-6)

    def test_predict_layer(self):
        hidden, inter, rank = 16, 32, 4
        num_layers = 2
        bank = PredictorBank(num_layers, hidden, inter, rank)

        x = mx.random.normal((1, hidden))
        indices = bank.predict_layer(0, x, threshold=0.5, min_neurons=4)
        mx.eval(indices)
        assert indices.shape[0] >= 4
        assert mx.all(indices < inter).item()

    def test_different_layers_give_different_predictions(self):
        """Each layer's predictor is independently initialized."""
        hidden, inter, rank = 16, 32, 4
        num_layers = 2
        bank = PredictorBank(num_layers, hidden, inter, rank)

        x = mx.random.normal((1, hidden))
        s0 = bank.predictors[0](x)
        s1 = bank.predictors[1](x)
        mx.eval(s0, s1)
        # Extremely unlikely to be identical given random init
        assert not mx.array_equal(s0, s1)
