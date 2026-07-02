"""Tests for prompt assembly + chat-template wrapping (pure Python, no torch)."""

import pytest

from text2sql.data.prompt import DEFAULT_SYSTEM_PROMPT, FewShotExample, PromptBuilder


class FakeStore:
    """Records the serialize kwargs it was called with."""

    def __init__(self):
        self.last_kwargs = None

    def serialize(self, db_id, **kwargs):
        self.last_kwargs = kwargs
        return f"Table: {db_id}\n  id (number)"

    def __contains__(self, db_id):
        return True


class FakeTokenizer:
    def __init__(self, eos_token_id=2):
        self.eos_token = "<eos>"
        self.eos_token_id = eos_token_id
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.padding_side = "right"
        self._vocab = {}
        self._next = 3

    def _id(self, tok):
        if tok not in self._vocab:
            self._vocab[tok] = self._next
            self._next += 1
        return self._vocab[tok]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = ["<bos>"]
        for m in messages:
            parts += [f"<|{m['role']}|>", m["content"]]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [self._id(t) for t in text.split()]}


DB, Q, SQL = "concert", "how many?", "SELECT count(*) FROM concert"


def test_build_messages_structure_and_content():
    b = PromptBuilder(FakeStore())  # tokenizer-free path
    msgs = b.build_messages(DB, Q)
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[-1]["content"]
    assert "### Database schema" in user
    assert "### Question" in user
    assert Q in user
    assert "Table: concert" in user


def test_system_prompt_optional():
    b = PromptBuilder(FakeStore(), system_prompt=None)
    assert [m["role"] for m in b.build_messages(DB, Q)] == ["user"]
    b2 = PromptBuilder(FakeStore())
    assert b2.build_messages(DB, Q)[0]["content"] == DEFAULT_SYSTEM_PROMPT


def test_serialize_kwargs_forwarded():
    store = FakeStore()
    b = PromptBuilder(store, serialize_kwargs={"include_foreign_keys": False})
    b.build_messages(DB, Q)
    assert store.last_kwargs == {"include_foreign_keys": False}


def test_few_shot_turns_and_content():
    b = PromptBuilder(FakeStore(), FakeTokenizer())
    shots = [FewShotExample("d1", "q1", "SELECT 1"), FewShotExample("d2", "q2", "SELECT 2")]
    roles = [m["role"] for m in b.build_messages(DB, Q, few_shot=shots)]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "SELECT 1" in b.render_inference_prompt(DB, Q, few_shot=shots)


def test_render_ends_with_generation_prompt():
    b = PromptBuilder(FakeStore(), FakeTokenizer())
    assert b.render_inference_prompt(DB, Q).rstrip().endswith("<|assistant|>")


def test_format_training_text_is_prompt_plus_sql_plus_eos():
    b = PromptBuilder(FakeStore(), FakeTokenizer())
    prompt = b.render_inference_prompt(DB, Q)
    text = b.format_training_text(DB, Q, SQL)
    assert text == prompt + SQL + "<eos>"


def test_render_requires_tokenizer():
    b = PromptBuilder(FakeStore())  # no tokenizer
    with pytest.raises(ValueError):
        b.render_inference_prompt(DB, Q)


def test_build_training_example_requires_eos():
    b = PromptBuilder(FakeStore(), FakeTokenizer(eos_token_id=None))
    with pytest.raises(ValueError):
        b.build_training_example(DB, Q, SQL)


def test_build_messages_deterministic():
    b = PromptBuilder(FakeStore())
    assert b.build_messages(DB, Q) == b.build_messages(DB, Q)