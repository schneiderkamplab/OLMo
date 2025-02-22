import logging
from pathlib import Path
from typing import Union

import yaml
import torch

from olmo.safetensors_util import safetensors_file_to_state_dict, state_dict_to_safetensors_file

logger = logging.getLogger(__name__)

def init_normal(
    tensor: torch.Tensor,
    std: float = 0.02,
    init_cutoff_factor: float = 3,
):
    cutoff_value = init_cutoff_factor * std
    torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)

def main(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    safe_tensors: bool = False,
    every: int = 1,
    prefix: str = "transformer.blocks.",
    reset: bool = False,
) -> None:
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_input = input_dir / "config.yaml"
    config_output = str(output_dir / "config.yaml")
    if safe_tensors:
        model_input = input_dir / "model.safetensors"
        model_output = str(output_dir / "model.safetensors")
    else:
        model_input = input_dir / "model.pt"
        model_output = str(output_dir / "model.pt")

    logger.info("Loading config from %s", config_input)
    with open(config_input, "rt") as f:
        config = yaml.safe_load(f)

    logger.info("Loading model state from %s", model_input)
    if safe_tensors:
        model_state_dict = safetensors_file_to_state_dict(model_input)
    else:
        model_state_dict = torch.load(model_input, map_location="cpu")

    mapping, added = compute_mapping_added(config["model"]["n_layers"], every)
    config["model"]["n_layers"] += len(added)
    new_model_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith(prefix):
            offset = prefix.count(".")
            parts = key.split(".")
            layer = int(parts[offset])
            for new_layer in mapping[layer]:
                new_key = ".".join(parts[:offset] + [str(new_layer)] + parts[offset+1:])
                new_value = value.clone()
                new_model_state_dict[new_key] = new_value
                print(f"Cloned {key} to {new_key}")
                if reset and new_layer in added:
                    init_normal(new_value)
                    print(f"Reinitialized {new_key}")
            model_state_dict[key] = None
        else:
            new_model_state_dict[key] = value
            print(f"Kept {key}")
    model_state_dict = new_model_state_dict

    logger.info("Saving config to %s", config_output)
    with open(config_output, "wt") as f:
        yaml.safe_dump(config, f)

    logger.info("Saving model state to %s", model_output)
    if safe_tensors:
        state_dict_to_safetensors_file(model_state_dict, model_output)
    else:
        torch.save(model_state_dict, model_output)
    del model_state_dict

def compute_mapping_added(n_layers, every):
    mapping = list(range(n_layers))
    inserts = list(range(every-1, n_layers, every))
    for i in reversed(inserts):
        mapping.insert(i, mapping[i])
    added = {i for i in range(1, len(mapping)) if mapping[i-1] == mapping[i]}
    _added = set(map(lambda x: x[0]+x[1], zip(inserts, range(1, len(inserts)+1))))
    assert added == _added, f"mapping = {mapping}\ninserts = {inserts}\nadded = {added}\n_added = {_added}"
    mapping = {i: [j for j in range(len(mapping)) if mapping[j] == i] for i in range(n_layers)}
    return mapping, added

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="unshard.py", description="Expand with additional layers")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--safe-tensors",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--prefix",
        default="transformer.blocks.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Reinitialize the weights of the added layers",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(
        args.input_dir,
        args.output_dir,
        safe_tensors=args.safe_tensors,
        every=args.every,
        prefix=args.prefix,
        reset=args.reset,
    )
