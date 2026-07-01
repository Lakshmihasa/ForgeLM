"""Prompt assembly and chat-template wrapping for text-to-SQL.

This module turns (schema, question) into the exact tokens a model sees, for
both training and inference. Its one non-negotiable job:

    the prompt a model is TRAINED on == the prompt it is EVALUATED on

Every model — base zero-shot, base few-shot, fine-tuned — routes through the
same `PromptBuilder`, so any change here changes the contract for all of them.
A mismatch between the training prompt and the eval prompt (wrong special
tokens, missing EOS, a stray scaffold string) is the most common silent killer
of a fine-tune, which is why prompt construction lives in one place and is
tested.

Design notes:
  * We use `tokenizer.apply_chat_template` rather than hand-writing special
    tokens, so switching base models (Llama-3 <-> Mistral) needs no code change.
  * The eval-spec `### SQL` scaffold is intentionally dropped: with chat models
    the assistant turn IS the SQL, and the chat template's generation prompt
    plays the scaffold's role. Keeping both would risk train/eval drift.
  * Few-shot examples become alternating user/assistant turns, each carrying its
    own schema (Spider few-shots come from other databases). Selection is the
    caller's job, so this module stays deterministic and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .schema import SchemaStore

if TYPE_CHECKING:  # avoid importing transformers just to load schemas / run tests
    from transformers import PreTrainedTokenizerBase

__all__ = ["FewShotExample", "PromptBuilder", "DEFAULT_SYSTEM_PROMPT"]


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert data analyst that translates natural-language questions "
    "into SQLite queries. Given a database schema and a question, respond with a "
    "single valid SQLite query that answers it. Output only the SQL query — no "
    "explanation, no comments, no markdown code fences."
)


@dataclass(frozen=True)
class FewShotExample:
    """One in-context example. Must come from a TRAIN database (never val/test),
    or the few-shot baseline leaks eval schemas."""

    db_id: str
    question: str
    sql: str


class PromptBuilder:
    """Builds chat messages and renders them into train / inference tensors.

    One builder is tied to one tokenizer + one serialization config. Construct it
    once per experiment and pass it everywhere.

    Parameters
    ----------
    schema_store   : loaded SchemaStore (source of schema text)
    tokenizer      : a HF tokenizer; required only for the render_* / build_*
                     methods, not for build_messages (which is tokenizer-free so
                     prompt *content* can be unit-tested without a model)
    system_prompt  : system turn; pass None to omit
    serialize_kwargs : forwarded to serialize_schema — this is the ablation knob
                       (include_foreign_keys, sample_values, ...). Freeze it for a
                       run so every model sees identical schema text.
    """

    def __init__(
        self,
        schema_store: SchemaStore,
        tokenizer: "PreTrainedTokenizerBase | None" = None,
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        serialize_kwargs: dict[str, Any] | None = None,
    ):
        self.schema_store = schema_store
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.serialize_kwargs = dict(serialize_kwargs or {})

    # ------------------------------------------------------------------ #
    # Message construction (tokenizer-free, deterministic)
    # ------------------------------------------------------------------ #
    def _user_content(self, db_id: str, question: str) -> str:
        schema_text = self.schema_store.serialize(db_id, **self.serialize_kwargs)
        return (
            "### Database schema\n"
            f"{schema_text}\n\n"
            "### Question\n"
            f"{question.strip()}"
        )

    def build_messages(
        self,
        db_id: str,
        question: str,
        few_shot: Sequence[FewShotExample] = (),
    ) -> list[dict[str, str]]:
        """Return the chat messages: [system?] + few-shot (user/assistant)* + user.

        The final user turn has no assistant turn — that's what the model fills in.
        """
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for ex in few_shot:
            messages.append(
                {"role": "user", "content": self._user_content(ex.db_id, ex.question)}
            )
            messages.append({"role": "assistant", "content": ex.sql.strip()})

        messages.append(
            {"role": "user", "content": self._user_content(db_id, question)}
        )
        return messages

    # ------------------------------------------------------------------ #
    # Inference rendering
    # ------------------------------------------------------------------ #
    def render_inference_prompt(
        self,
        db_id: str,
        question: str,
        few_shot: Sequence[FewShotExample] = (),
    ) -> str:
        """The exact prompt string fed to the model at eval/inference time.

        Ends with the model's generation prompt (assistant header), so the model
        continues with the SQL.
        """
        self._require_tokenizer()
        messages = self.build_messages(db_id, question, few_shot)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------ #
    # Training example construction (with loss masking)
    # ------------------------------------------------------------------ #
    def build_training_example(
        self,
        db_id: str,
        question: str,
        target_sql: str,
        few_shot: Sequence[FewShotExample] = (),
        *,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        """Tokenize one supervised example with the prompt masked out.

        Returns input_ids / attention_mask / labels, where labels are -100 on the
        prompt and equal to the token ids on the completion (target SQL + EOS).
        Only the SQL contributes to the loss — the model is not trained to
        reproduce the schema.

        The prompt tokens here are byte-identical to `render_inference_prompt`,
        which is what guarantees train/eval parity. Prompt and completion are
        tokenized separately and concatenated so the mask boundary is exact
        (avoids the tokenizer merging the last prompt token with the first
        completion token).
        """
        self._require_tokenizer()
        eos_id = self._eos_id()

        prompt_text = self.render_inference_prompt(db_id, question, few_shot)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(
            target_sql.strip(), add_special_tokens=False
        )["input_ids"] + [eos_id]

        if max_length is not None and len(prompt_ids) + len(completion_ids) > max_length:
            # Preserve the whole completion; drop earliest prompt tokens
            # (oldest few-shot / start of schema) from the left.
            keep_prompt = max_length - len(completion_ids)
            if keep_prompt <= 0:
                # Completion alone already too long; truncate it from the right.
                completion_ids = completion_ids[: max_length - 1] + [eos_id]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-keep_prompt:]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + list(completion_ids)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def format_training_text(
        self,
        db_id: str,
        question: str,
        target_sql: str,
        few_shot: Sequence[FewShotExample] = (),
    ) -> str:
        """Full training string (prompt + SQL + EOS) as text.

        Convenience for TRL `SFTTrainer(dataset_text_field="text")`. NOTE: this
        does NOT mask the prompt — pair it with a completion-only collator, or
        prefer `build_training_example` which masks for you.
        """
        self._require_tokenizer()
        prompt_text = self.render_inference_prompt(db_id, question, few_shot)
        return prompt_text + target_sql.strip() + (self.tokenizer.eos_token or "")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _require_tokenizer(self) -> None:
        if self.tokenizer is None:
            raise ValueError(
                "This method needs a tokenizer. Construct PromptBuilder with one, "
                "or use build_messages() for tokenizer-free prompt content."
            )

    def _eos_id(self) -> int:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError(
                "Tokenizer has no eos_token_id; set it before building training "
                "examples (the model must learn to stop after the query)."
            )
        return eos


# --------------------------------------------------------------------------- #
# Manual sanity check:
#   python -m text2sql.data.prompt <tables.json> [db_id] [model_name]
# Without a model_name, prints the tokenizer-free message content.
# With one, also prints the rendered chat-template prompt.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python prompt.py <tables.json> [db_id] [model_name]")
        raise SystemExit(1)

    store = SchemaStore(sys.argv[1])
    db_id = sys.argv[2] if len(sys.argv) > 2 else store.db_ids()[0]
    question = "How many rows are in the first table?"

    builder = PromptBuilder(store)
    print("=== messages (tokenizer-free) ===")
    for m in builder.build_messages(db_id, question):
        print(f"\n[{m['role']}]\n{m['content']}")

    if len(sys.argv) > 3:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(sys.argv[3])
        builder = PromptBuilder(store, tok)
        print("\n\n=== rendered inference prompt ===")
        print(builder.render_inference_prompt(db_id, question))