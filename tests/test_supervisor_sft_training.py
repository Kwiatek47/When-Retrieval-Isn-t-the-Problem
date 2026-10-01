from __future__ import annotations

import unittest


class _FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        self.assert_tokenize = tokenize
        tokens: list[int] = []
        role_tokens = {"system": 10, "user": 20, "assistant": 30}
        for message in messages:
            tokens.extend([role_tokens[message["role"]], len(message["content"])])
        if add_generation_prompt:
            tokens.append(30)
        else:
            tokens.append(self.eos_token_id)
        return tokens


class SupervisorSftTrainingTests(unittest.TestCase):
    def test_encode_masks_everything_before_assistant_answer(self) -> None:
        from scripts.sft.train_qwen14b_supervisor_unsloth import encode_chat_example

        tokenizer = _FakeTokenizer()
        encoded = encode_chat_example(
            {
                "messages": [
                    {"role": "system", "content": "json"},
                    {"role": "user", "content": "case"},
                    {"role": "assistant", "content": '{"final_label":"yes"}'},
                ]
            },
            tokenizer=tokenizer,
            max_seq_length=64,
        )

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "json"},
                {"role": "user", "content": "case"},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        self.assertEqual(encoded["labels"][: len(prompt)], [-100] * len(prompt))
        self.assertEqual(encoded["labels"][len(prompt) :], encoded["input_ids"][len(prompt) :])
        self.assertTrue(any(label != -100 for label in encoded["labels"]))

    def test_encode_rejects_non_assistant_final_turn(self) -> None:
        from scripts.sft.train_qwen14b_supervisor_unsloth import encode_chat_example

        with self.assertRaisesRegex(ValueError, "assistant"):
            encode_chat_example(
                {"messages": [{"role": "user", "content": "case"}]},
                tokenizer=_FakeTokenizer(),
                max_seq_length=64,
            )

    def test_encode_rejects_truncated_answer(self) -> None:
        from scripts.sft.train_qwen14b_supervisor_unsloth import encode_chat_example

        with self.assertRaisesRegex(ValueError, "answer"):
            encode_chat_example(
                {
                    "messages": [
                        {"role": "system", "content": "json"},
                        {"role": "user", "content": "case"},
                        {"role": "assistant", "content": '{"final_label":"maybe"}'},
                    ]
                },
                tokenizer=_FakeTokenizer(),
                max_seq_length=5,
            )

    def test_default_config_matches_single_a40_plan(self) -> None:
        from scripts.sft.train_qwen14b_supervisor_unsloth import TrainingConfig

        config = TrainingConfig()
        self.assertEqual(config.base_model, "unsloth/Qwen2.5-14B-Instruct-bnb-4bit")
        self.assertEqual(config.max_seq_length, 4096)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.gradient_accumulation_steps, 16)
        self.assertEqual(config.lora_r, 32)
        self.assertEqual(config.lora_alpha, 64)
        self.assertEqual(config.seed, 47)


if __name__ == "__main__":
    unittest.main()
