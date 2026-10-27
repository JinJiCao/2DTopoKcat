
import os
import json
from pathlib import Path

import torch
from tqdm import tqdm
from utils.protein_init import seq_feature


ESMC6B_MODEL_ID = "Biohub/ESMC-6B"
DEFAULT_ESMC6B_MODEL_PATHS = (
    str(Path(__file__).resolve().parents[1] / "pretrained" / "ESMC-6B"),
)
ESMC6B_CACHE_VERSION = 2
STANDARD_AA_WITH_UNKNOWN = set("ACDEFGHIKLMNPQRSTVWYX")


def _sanitize_sequence(sequence):

    sequence = str(sequence).upper()
    return "".join(
        residue if residue in STANDARD_AA_WITH_UNKNOWN else "X"
        for residue in sequence
    )


def _resolve_model_path():

    configured = os.environ.get("ESMC6B_MODEL_PATH")
    candidates_to_check = [configured] if configured else DEFAULT_ESMC6B_MODEL_PATHS
    for candidate in candidates_to_check:
        configured_path = Path(candidate).expanduser()
        if (configured_path / "config.json").is_file():
            return str(configured_path)
        snapshots = configured_path / "snapshots"
        if snapshots.is_dir():
            candidates = [
                path.parent for path in snapshots.glob("*/config.json")
                if path.is_file()
            ]
            if candidates:
                return str(max(candidates, key=lambda path: path.stat().st_mtime))

    raise FileNotFoundError(
        "Cannot find ESMC-6B config.json under "
        f"{candidates_to_check}. Set ESMC6B_MODEL_PATH to the directory containing "
        "config.json or to a Hugging Face cache directory containing snapshots/."
    )


def _load_hf_esmc6b_weights(model, model_path, device):






    from safetensors.torch import load_file

    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sharded checkpoint index: {index_path}")

    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    native_keys = set(model.state_dict().keys())
    loaded_keys = set()
    ignored_keys = []

    for shard_name in sorted(set(weight_map.values())):
        shard_path = model_path / shard_name
        print(f"Loading ESMC-6B shard {shard_name} ...")
                                                                           
                                                                             
                                                                          
        shard = load_file(str(shard_path), device="cpu")
        mapped = {}
        for source_key, tensor in shard.items():
            if source_key.startswith("esmc."):
                target_key = source_key[len("esmc."):]
            elif source_key.startswith("lm_head."):
                target_key = "sequence_head." + source_key[len("lm_head."):]
            else:
                target_key = source_key

                                                                          
                                                                        
            replacements = {
                ".attn.layernorm_qkv.layer_norm_weight": ".attn.layernorm_qkv.0.weight",
                ".attn.layernorm_qkv.layer_norm_bias": ".attn.layernorm_qkv.0.bias",
                ".attn.layernorm_qkv.weight": ".attn.layernorm_qkv.1.weight",
                ".ffn.layer_norm_weight": ".ffn.0.weight",
                ".ffn.layer_norm_bias": ".ffn.0.bias",
                ".ffn.fc1_weight": ".ffn.1.weight",
                ".ffn.fc2_weight": ".ffn.3.weight",
            }
            for hf_suffix, native_suffix in replacements.items():
                if target_key.endswith(hf_suffix):
                    target_key = target_key[:-len(hf_suffix)] + native_suffix
                    break

            if target_key in native_keys:
                mapped[target_key] = tensor.to(
                    device=device, dtype=torch.bfloat16, non_blocking=False
                )
                loaded_keys.add(target_key)
            else:
                ignored_keys.append((source_key, target_key))

        model.load_state_dict(mapped, strict=False, assign=True)
        del shard, mapped

    missing = sorted(native_keys - loaded_keys)
    meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if missing or meta_parameters:
        raise RuntimeError(
            "ESMC-6B checkpoint mapping incomplete. "
            f"missing_keys={missing[:20]}, meta_parameters={meta_parameters[:20]}, "
            f"ignored_keys={ignored_keys[:20]}"
        )
    if ignored_keys:
        print(f"Ignored {len(ignored_keys)} checkpoint tensors not used by native ESMC")
    return model


def _inspect_checkpoint_architecture(model_path):

    from safetensors import safe_open

    model_path = Path(model_path)
    with (model_path / "model.safetensors.index.json").open(
        "r", encoding="utf-8"
    ) as handle:
        weight_map = json.load(handle)["weight_map"]

    prefix = "esmc.transformer.blocks."
    block_ids = {
        int(key[len(prefix):].split(".", 1)[0])
        for key in weight_map if key.startswith(prefix)
    }
    if not block_ids or block_ids != set(range(max(block_ids) + 1)):
        raise RuntimeError(f"Non-contiguous ESMC block ids: {sorted(block_ids)}")
    num_layers = max(block_ids) + 1

    embed_key = "esmc.embed.weight"
    embed_shard = model_path / weight_map[embed_key]
    with safe_open(str(embed_shard), framework="pt", device="cpu") as handle:
        hidden_dim = int(handle.get_slice(embed_key).get_shape()[1])

                                                                              
    if hidden_dim % 64:
        raise RuntimeError(f"Cannot infer ESMC attention heads from D={hidden_dim}")
    num_heads = hidden_dim // 64
    return num_layers, hidden_dim, num_heads


def _select_relative_layers(num_layers, count=4):

    if num_layers < count:
        raise ValueError(f"Need at least {count} transformer blocks, got {num_layers}")
    selected = []
    for fraction in (0.25, 0.50, 0.75, 1.00):
        layer = min(num_layers, max(1, round(num_layers * fraction)))
        if layer not in selected:
            selected.append(layer)
    for layer in range(1, num_layers + 1):
        if len(selected) == count:
            break
        if layer not in selected:
            selected.append(layer)
    return sorted(selected)


def _extract_selected_hidden_states(
    model, sequence, selected_layers, chunk_size=1022, overlap=256
):

    sequence = _sanitize_sequence(sequence)
    step = chunk_size - overlap
    starts = [0] if len(sequence) <= chunk_size else list(range(0, len(sequence), step))
    chunks = []
    for start in starts:
        end = min(start + chunk_size, len(sequence))
        chunks.append((start, end))
        if end == len(sequence):
            break

    merged = None
    counts = torch.zeros(len(sequence), 1, 1)
    zero_based = [layer - 1 for layer in selected_layers]
    for start, end in chunks:
        from esm.sdk.api import ESMProtein

        protein_tensor = model.encode(ESMProtein(sequence=sequence[start:end]))
        sequence_tokens = protein_tensor.sequence.unsqueeze(0)
                                                                                 
        if sequence_tokens.size(1) < (end - start) + 2:
            raise RuntimeError("ESMC tokenizer did not produce BOS/residues/EOS")
        with torch.inference_mode():
            output = model(sequence_tokens=sequence_tokens)
        hidden_states = output.hidden_states
        if isinstance(hidden_states, (tuple, list)):
            selected = torch.stack([hidden_states[i] for i in zero_based], dim=0)
        else:
            if hidden_states.size(0) != len(model.transformer.blocks):
                raise RuntimeError(f"Unexpected hidden-state shape {hidden_states.shape}")
            selected = hidden_states[zero_based]
                                                                                
                                                                                 
        residue_embedding = selected[:, 0, 1:1 + end - start].permute(1, 0, 2)
        residue_embedding = residue_embedding.float().cpu()
        if residue_embedding.shape[:2] != (end - start, len(selected_layers)):
            raise RuntimeError(
                "ESMC token/residue mismatch: "
                f"shape={tuple(residue_embedding.shape)}, residues={end-start}"
            )
        if merged is None:
            merged = torch.zeros(
                len(sequence), len(selected_layers), residue_embedding.size(-1)
            )
        merged[start:end] += residue_embedding
        counts[start:end] += 1

    return merged / counts.clamp_min(1.0)


def protein_init(seqs, chunk_size=1022, overlap=256):

    from accelerate import init_empty_weights
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    if not torch.cuda.is_available():
        raise RuntimeError("Local ESMC-6B extraction requires a CUDA GPU")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    model_path = _resolve_model_path()
    num_layers, hidden_dim, num_heads = _inspect_checkpoint_architecture(model_path)
    selected_layers = _select_relative_layers(num_layers)
    print(f"Loading local ESMC-6B from {model_path}")
    print(
        f"Checkpoint architecture: N={num_layers}, D={hidden_dim}, "
        f"heads={num_heads}; selected one-based layers={selected_layers}"
    )
    device = torch.device("cuda")
    with init_empty_weights():
        model = ESMC(
            d_model=hidden_dim,
            n_heads=num_heads,
            n_layers=num_layers,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=True,
        ).eval()
    model = _load_hf_esmc6b_weights(model, model_path, device)
                                                                           
                                                                            
                                                   
    model = model.to(device=device, dtype=torch.bfloat16)

    result_dict = {}
    for seq in tqdm(seqs, desc="ESMC-6B four-layer features"):
        clean_seq = _sanitize_sequence(seq)
        replaced_residues = sorted({
            residue for residue in str(seq).upper()
            if residue not in STANDARD_AA_WITH_UNKNOWN
        })
        representations = _extract_selected_hidden_states(
            model, seq, selected_layers, chunk_size=chunk_size, overlap=overlap
        )
        result_dict[seq] = {
            "seq": seq,
            "seq_feat": torch.from_numpy(seq_feature(clean_seq)).float(),
            "esmc_multi_layer": representations.to(torch.bfloat16),
            "selected_layers": selected_layers,
            "num_transformer_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_nodes": len(seq),
            "num_pos": torch.arange(len(seq)).reshape(-1, 1),
            "encoder": ESMC6B_MODEL_ID,
            "encoder_path": model_path,
            "hidden_states_include_embedding": False,
            "special_token_handling": "removed_BOS_EOS_no_padding_single_sequence",
            "nonstandard_residues_replaced_with_X": replaced_residues,
            "cache_version": ESMC6B_CACHE_VERSION,
        }

    del model
    torch.cuda.empty_cache()
    return result_dict
