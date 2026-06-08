from olmlx.engine.model_manager import LoadedModel, _is_cross_encoder_config


def test_is_cross_encoder_config_true_for_sequence_classification():
    cfg = {
        "architectures": ["XLMRobertaForSequenceClassification"],
        "model_type": "xlm-roberta",
    }
    assert _is_cross_encoder_config(cfg) is True


def test_is_cross_encoder_config_false_for_plain_lm():
    cfg = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}
    assert _is_cross_encoder_config(cfg) is False


def test_is_cross_encoder_config_false_when_no_architectures():
    assert _is_cross_encoder_config({"model_type": "bert"}) is False


def test_is_cross_encoder_config_false_for_non_roberta_classifier():
    # A plain BERT/DistilBERT text classifier ends in ForSequenceClassification
    # but is NOT a supported reranker — the encoder only handles XLM-RoBERTa.
    cfg = {"architectures": ["BertForSequenceClassification"], "model_type": "bert"}
    assert _is_cross_encoder_config(cfg) is False


def test_is_cross_encoder_config_true_for_roberta_family():
    cfg = {"architectures": ["RobertaForSequenceClassification"], "model_type": "roberta"}
    assert _is_cross_encoder_config(cfg) is True


def test_loaded_model_is_reranker_field_defaults_false():
    lm = LoadedModel(name="m", hf_path="m", model=object(), tokenizer=object())
    assert lm.is_reranker is False


def test_loaded_model_accepts_is_reranker():
    lm = LoadedModel(
        name="bge", hf_path="bge", model=object(), tokenizer=object(), is_reranker=True
    )
    assert lm.is_reranker is True
