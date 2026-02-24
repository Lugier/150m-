# Plan-Compliance: Instruction SFT + RL & Best-in-Class 150M

## Instruction SFT + RL Pipeline (instruction_sft_+_rl_pipeline_32501a34)

| Plan-Anforderung | Umsetzung | Status |
|------------------|-----------|--------|
| **§1 Chat-Format** | `data/chat_format.py`: ChatML (`format_chat_message`, `format_chat_history`), `parse_chat_to_message_spans` für Assistant-Only-Labels; `data/chat_template.yaml` | ✅ |
| **§2 Instruction-Daten** | `data/instruction_data.py`: `evolve_instruction`, `generate_teacher_response`, Teacher-API (Ollama/OpenAI); `data/generate_instruction_data.py` → `instruction_sft.jsonl` | ✅ |
| **§3 SFT-Dataloader** | `data/sft_dataloader.py`: JSONL → ChatML, `parse_chat_to_message_spans` → Labels nur auf Assistant (-100 sonst), Padding, `--instruction_data` optional | ✅ |
| **§4 SFT-Training** | `training/sft_train.py`: Pre-Train-Checkpoint, Assistant-Only-Loss, Config `config_sft.yaml`, `--checkpoint`, `--instruction_data`, `--tokenizer_path`, `--device`, EMA, Gradient Clipping; `final_sft.pt` | ✅ |
| **§5 Execution-RL** | `post_training/execution_reward.py`: Syntax (-1.0), Sandbox/Runtime (-0.5/0.0), Tests (+1.0/-0.5); Tuple-Entpackung von `run_tests_in_sandbox`. `training/rl_train.py`: GRPO, SFT-Checkpoint, Tokenizer-Load, RL-Daten aus JSONL oder built-in | ✅ |
| **§6 Pipeline** | Pre-Train → SFT → RL; Modell ≤250M (config gesteuert) | ✅ |
| **§7 Dateien** | Alle genannten Dateien vorhanden (chat_format, instruction_data, generate_instruction_data, sft_dataloader, config_sft, sft_train, execution_reward, rl_train) | ✅ |
| **§8 Inferenz** | `inference/run_chat.py`: ChatML-Prompt, nur Assistant-Ausgabe; Tokenizer-Pfad Pflicht, **ohne Fallback** | ✅ |

## Best-in-Class 150M (best-in-class_150m_code_llm_65ad1b0a)

Alle Säulen des Plans sind **verbindlich integriert** (kein optionaler Research-Track):

| Säule | Umsetzung |
|-------|-----------|
| **BLT** | `model/blt.py` (BytePatchEncoder/Decoder); `model/config.py` `use_blt`; `gpt.py`: Byte-Embed + Latent-Transformer + Byte-Head (256 vocab) |
| **Mamba-2-Hybrid** | `model/mamba_hybrid.py` (43/7/50); `gpt.py` baut Blocks via `make_mamba_hybrid_layer` wenn `use_mamba_hybrid` |
| **LEAM++** | `model/leam.py` (LEAMGrammarConstrainer); Inferenz: `run_chat.py` / `run_torch.py` wenden `constrain_logits` an wenn `use_leam` |
| **BitNet / L-MTP** | `model/gpt.py` BitLinear, mtp_heads; Config `use_bitnet`, `mtp_n` |
| **CODA / CodeDenoise** | `data/coda.py`, `data/code_denoise.py`; `data/prepare_data.py` Stage-2-Stream: `run_code_denoise` + `coda_mutation_rate` aus `config_data.yaml` |
| **CodeRL+** | `post_training/coderl_plus.py`; `training/rl_train.py`: optional `reference_solution`/`target_code` in RL-JSONL → blended Reward (Execution + Semantics-Match) |
| **Test-Time Evolution** | `inference/test_time_evolution.py`: S* (`s_star_select`, `s_star_generate`), AB-MCTS (`ab_mcts_score`), DaJ (`daj_judge`), PoT (`pot_update_hook`); Sandbox-Anbindung über `eval_repair.run_tests_in_sandbox` |

- **Daten & Config:** Drei Stages in `data/config_data.yaml`, RegMix, CODA/CodeDenoise in Stage-2-Pipeline; `config_train.yaml` / `config_sft.yaml`: `use_mamba_hybrid`, `use_blt`, `use_leam`.
- **Training:** `train.py`, `sft_train.py`, `rl_train.py` laden alle ModelConfig-Optionen (use_bitnet, use_mamba_hybrid, use_blt, use_leam).
- **Inferenz:** `run_chat.py`, `run_torch.py` mit LEAM-Option; Test-Time Evolution über `inference/test_time_evolution.py` aufrufbar.

## Production-Readiness (gemäß Instructions)

- **Keine Fallbacks:** Alle Dummy-/Character-Level-/Synthetic-Fallbacks entfernt. Tokenizer, RL-Daten, SFT-Daten und Pre-Train-Dataloader sind **Pflicht**; bei fehlenden Ressourcen erfolgt klarer Abbruch mit `FileNotFoundError`/`ValueError`.
- **RL:** `--rl-data` ist **required**; Tokenizer muss unter `data/tokenizer` (oder `--tokenizer`) existieren.
- **SFT:** Dataloader und Instruction-JSONL sind Pflicht; Tokenizer-Pfad und JSONL müssen existieren.
- **Pre-Train:** `get_training_dataloader` und gültiger Dataloader Pflicht; kein Fallback-Batch.
- **Inferenz (run_chat, run_torch):** Tokenizer-Pfad muss existieren; kein Fallback-Tokenizer.
- **Tokenizer-Loader:** `load_tokenizer_for_training` nutzt nur konfigurierten Pfad bzw. `model/vocab`; kein HF-Auto-Fallback.
- **Execution Reward:** Korrekte Verwendung von `run_tests_in_sandbox` (Tupel entpackt).
