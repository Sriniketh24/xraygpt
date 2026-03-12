"""
Report decoder module using GPT-2 for autoregressive text generation.

Takes visual prefix tokens + text tokens and generates radiology reports.
We use HuggingFace's GPT-2 implementation and modify the forward pass
to accept prepended visual tokens.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class ReportDecoder(nn.Module):
    """
    GPT-2 based report decoder with visual prefix support.

    During training:
        Input = [visual_prefix_tokens | report_tokens]
        Loss is computed only on the report tokens (not the visual prefix).

    During inference:
        Input = [visual_prefix_tokens]
        Autoregressively generate report tokens.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        freeze_embeddings: bool = False,
    ):
        super().__init__()

        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # GPT-2 Small: hidden_size=768, n_layer=12, n_head=12
        self.hidden_size = self.gpt2.config.n_embd

        if freeze_embeddings:
            for param in self.gpt2.transformer.wte.parameters():
                param.requires_grad = False
            for param in self.gpt2.transformer.wpe.parameters():
                param.requires_grad = False

    def forward(
        self,
        prefix_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with visual prefix prepended to text embeddings.

        Args:
            prefix_embeds: (B, num_prefix, hidden_size) visual prefix tokens
            input_ids: (B, seq_len) tokenized report
            attention_mask: (B, seq_len) attention mask for report tokens

        Returns:
            dict with 'loss' and 'logits'
        """
        batch_size = input_ids.shape[0]
        prefix_len = prefix_embeds.shape[1]

        # Get text embeddings from GPT-2's embedding layer
        text_embeds = self.gpt2.transformer.wte(input_ids)

        # Add positional embeddings to text
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            prefix_len, prefix_len + seq_len, device=input_ids.device
        )
        position_embeds = self.gpt2.transformer.wpe(position_ids)
        text_embeds = text_embeds + position_embeds

        # Add positional embeddings to prefix
        prefix_position_ids = torch.arange(prefix_len, device=input_ids.device)
        prefix_position_embeds = self.gpt2.transformer.wpe(prefix_position_ids)
        prefix_embeds = prefix_embeds + prefix_position_embeds

        # Concatenate: [visual_prefix | text_tokens]
        combined_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        # Build attention mask: all 1s for prefix, use provided mask for text
        prefix_mask = torch.ones(
            batch_size, prefix_len, dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Build labels: -100 for prefix (ignored in loss), input_ids for text
        # Shift is handled internally by GPT-2's loss computation
        prefix_labels = torch.full(
            (batch_size, prefix_len), -100, dtype=input_ids.dtype, device=input_ids.device
        )
        combined_labels = torch.cat([prefix_labels, input_ids], dim=1)
        # Mask padding tokens in labels
        combined_labels[combined_labels == self.tokenizer.pad_token_id] = -100

        # Forward through GPT-2 (skip embedding layers since we prepared embeds)
        outputs = self.gpt2(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        prefix_embeds: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        beam_size: int = 1,
        repetition_penalty: float = 1.2,
    ) -> list[str]:
        """
        Generate reports from visual prefix tokens.

        Args:
            prefix_embeds: (B, num_prefix, hidden_size) visual prefix tokens
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            beam_size: beam search width (1 = greedy)
            repetition_penalty: penalty for repeated tokens

        Returns:
            List of generated report strings
        """
        batch_size = prefix_embeds.shape[0]
        prefix_len = prefix_embeds.shape[1]
        device = prefix_embeds.device

        # Add positional embeddings to prefix
        prefix_position_ids = torch.arange(prefix_len, device=device)
        prefix_position_embeds = self.gpt2.transformer.wpe(prefix_position_ids)
        prefix_embeds = prefix_embeds + prefix_position_embeds

        # Start with BOS token
        bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        generated_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )

        # Get initial text embedding
        text_embeds = self.gpt2.transformer.wte(generated_ids)
        pos_offset = prefix_len
        position_ids = torch.tensor([[pos_offset]], device=device).expand(batch_size, -1)
        text_embeds = text_embeds + self.gpt2.transformer.wpe(position_ids)

        combined_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        # Autoregressive generation
        past_key_values = None
        all_generated = generated_ids

        for step in range(max_new_tokens):
            if past_key_values is None:
                outputs = self.gpt2(
                    inputs_embeds=combined_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # Only pass the new token embedding
                new_token_embed = self.gpt2.transformer.wte(generated_ids[:, -1:])
                pos_id = torch.tensor(
                    [[pos_offset + step]], device=device
                ).expand(batch_size, -1)
                new_token_embed = new_token_embed + self.gpt2.transformer.wpe(pos_id)

                outputs = self.gpt2(
                    inputs_embeds=new_token_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # (B, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(all_generated[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Greedy or sampling
            if beam_size <= 1 and top_p >= 1.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs - probs > top_p
                sorted_logits[sorted_mask] = float("-inf")

                probs = torch.softmax(sorted_logits, dim=-1)
                next_token_sorted = torch.multinomial(probs, num_samples=1)
                next_token = sorted_indices.gather(1, next_token_sorted)

            all_generated = torch.cat([all_generated, next_token], dim=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop if all sequences have generated EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break

        # Decode generated tokens
        reports = []
        for i in range(batch_size):
            text = self.tokenizer.decode(
                all_generated[i, 1:],  # skip BOS
                skip_special_tokens=True,
            )
            reports.append(text.strip())

        return reports
