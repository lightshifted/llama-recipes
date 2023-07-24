# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
from .utils import Concatenator

def remove_start_token(text):
      return text.replace("<START>", "")

def get_preprocessed_roleplay(dataset_config, tokenizer, split):

  dataset = datasets.load_dataset("hedronstone/gptteacher_role_play-lmgym", split=split)

  prompt = (
      f"Have an engaging conversation while in this persona: \n{{persona}}\n---Start:\n{{opening}}{{eos_token}}"
  )

  def apply_prompt_template(sample):
    return {
        "text": prompt.format(
            persona=remove_start_token(sample["input_text"]),
            opening=sample["output_text"],
            eos_token=tokenizer.eos_token,
        )
    }

  dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

  dataset = dataset.map(
      lambda sample: tokenizer(sample["text"]),
      batched=True,
      remove_columns=list(dataset.features),
  ).map(Concatenator(), batched=True)
  return dataset