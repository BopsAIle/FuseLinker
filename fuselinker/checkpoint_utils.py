import torch
from typing import Any, Dict


def load_checkpoint_bundle(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint bundle from disk."""
    return torch.load(path, map_location=map_location)


def infer_checkpoint_specs(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Infer simple metadata from a checkpoint bundle."""
    specs: Dict[str, Any] = {}
    if not isinstance(bundle, dict):
        return specs

    if "model_module" in bundle:
        specs["model_module"] = bundle["model_module"]
    if "hidden_dim" in bundle:
        specs["hidden_dim"] = bundle["hidden_dim"]
    if "embedding_file" in bundle:
        specs["embedding_file"] = bundle["embedding_file"]
    if "domain_file" in bundle:
        specs["domain_file"] = bundle["domain_file"]

    state_dict = bundle.get("state_dict", bundle)
    if isinstance(state_dict, dict):
        if "model_module" not in specs:
            specs.setdefault("model_module", "model_base4")
        if "hidden_dim" not in specs:
            for name, param in state_dict.items():
                if name.endswith("poincare_to_euclidean.weight"):
                    specs["hidden_dim"] = param.shape[0]
                    break
                if name.endswith("autoencoder.encoder.0.weight"):
                    specs["hidden_dim"] = param.shape[0]
                    break
    return specs


def validate_eval_against_checkpoint(
    ckpt_specs: Dict[str, Any],
    text_embeddings: Any,
    ontology_embeddings: Any,
    embedding_file_basename: str,
    domain_file: str,
    n_hidden: int,
) -> None:
    """Validate that evaluation settings match checkpoint metadata."""
    if not ckpt_specs:
        return

    if "hidden_dim" in ckpt_specs and ckpt_specs["hidden_dim"] != n_hidden:
        print(
            f"[WARN] hidden_dim mismatch: checkpoint={ckpt_specs['hidden_dim']} vs eval={n_hidden}"
        )
    if "embedding_file" in ckpt_specs and ckpt_specs["embedding_file"] != embedding_file_basename:
        print(
            f"[WARN] embedding file mismatch: checkpoint={ckpt_specs['embedding_file']} vs eval={embedding_file_basename}"
        )
    if "domain_file" in ckpt_specs and ckpt_specs["domain_file"] != domain_file:
        print(
            f"[WARN] domain file mismatch: checkpoint={ckpt_specs['domain_file']} vs eval={domain_file}"
        )
