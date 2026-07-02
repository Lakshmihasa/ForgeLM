"""The parity guarantee, as a CI check.

The project's central claim is that a model is TRAINED on the same prompt it is
EVALUATED on. That property lives in PromptBuilder: build_training_example builds
its prompt via the same render_inference_prompt the harness uses. These tests
pin it down so a future refactor that duplicates the prompt logic (and lets train
and eval drift) fails loudly.

Hermetic: a fake tokenizer + fake schema store, no torch, no model download.
"""

from text2sql.data.format import Example
from text2sql.data.prompt import FewShotExample, PromptBuilder


class FakeStore:
    """Stand-in for SchemaStore: PromptBuilder only calls .serialize(db_id, **kw)."""

    def serialize(self, db_id, **kwargs):
        return f"Table: {db_id}\n  id (number)\n  name (text)"


class FakeTokenizer:
    """Deterministic word-level tokenizer with a chat template. Enough surface
    for render_inference_prompt / build_training_example / format_training_text."""

    def __init__(self):
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.padding_side = "right"
        self._vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        self._next = 3

    def _id(self, tok):
        if tok not in self._vocab:
            self._vocab[tok] = self._next
            self._next += 1
        return self._vocab[tok]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = ["<bos>"]
        for m in messages:
            parts.append(f"<|{m['role']}|>")
            parts.append(m["content"])
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [self._id(t) for t in text.split()]}


def _builder():
    return PromptBuilder(FakeStore(), FakeTokenizer(), system_prompt="you are a sql model")


DB, Q, SQL = "concert", "how many rows?", "SELECT count(*) FROM concert"


def test_training_prompt_matches_inference_prompt():
    """The heart of it: the training example's prompt tokens are exactly the
    inference prompt tokens."""
    b = _builder()
    prompt_text = b.render_inference_prompt(DB, Q)
    prompt_ids = b.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    ex = b.build_training_example(DB, Q, SQL)
    assert ex["input_ids"][: len(prompt_ids)] == prompt_ids


def test_prompt_is_masked_completion_is_supervised():
    b = _builder()
    prompt_ids = b.tokenizer(b.render_inference_prompt(DB, Q), add_special_tokens=False)["input_ids"]
    ex = b.build_training_example(DB, Q, SQL)

    n = len(prompt_ids)
    # prompt region: all -100
    assert ex["labels"][:n] == [-100] * n
    # completion region: supervised (equals the input ids there)
    assert ex["labels"][n:] == ex["input_ids"][n:]
    assert all(l != -100 for l in ex["labels"][n:])


def test_completion_ends_with_eos():
    b = _builder()
    ex = b.build_training_example(DB, Q, SQL)
    assert ex["input_ids"][-1] == b.tokenizer.eos_token_id
    assert ex["labels"][-1] == b.tokenizer.eos_token_id


def test_attention_mask_matches_length():
    b = _builder()
    ex = b.build_training_example(DB, Q, SQL)
    assert len(ex["attention_mask"]) == len(ex["input_ids"])
    assert set(ex["attention_mask"]) == {1}


def test_rendering_is_deterministic():
    b = _builder()
    assert b.render_inference_prompt(DB, Q) == b.render_inference_prompt(DB, Q)


def test_max_length_preserves_completion():
    b = _builder()
    full = b.build_training_example(DB, Q, SQL)
    completion_len = sum(1 for x in full["labels"] if x != -100)

    capped = b.build_training_example(DB, Q, SQL, max_length=completion_len + 3)
    assert len(capped["input_ids"]) <= completion_len + 3
    # the whole completion (SQL + EOS) survives truncation
    assert capped["input_ids"][-1] == b.tokenizer.eos_token_id
    assert sum(1 for x in capped["labels"] if x != -100) == completion_len


def test_few_shot_builds_alternating_turns():
    b = _builder()
    shots = [
        FewShotExample(db_id="d1", question="q1", sql="SELECT 1"),
        FewShotExample(db_id="d2", question="q2", sql="SELECT 2"),
    ]
    messages = b.build_messages(DB, Q, few_shot=shots)
    roles = [m["role"] for m in messages]
    # system, then (user, assistant) per shot, then the final user turn
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    # few-shot SQL appears in the rendered prompt
    assert "SELECT 1" in b.render_inference_prompt(DB, Q, few_shot=shots)


def test_examples_flow_through_from_dataset_records():
    """Example (from format.read_jsonl) fields feed straight into the builder."""
    ex = Example(id="abc", db_id=DB, question=Q, query=SQL)
    b = _builder()
    enc = b.build_training_example(ex.db_id, ex.question, ex.query)
    assert enc["input_ids"] and enc["labels"][-1] == b.tokenizer.eos_token_id