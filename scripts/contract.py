import logging
from pathlib import Path
from typing import Union

import yaml
import torch

from olmo.safetensors_util import safetensors_file_to_state_dict, state_dict_to_safetensors_file

logger = logging.getLogger(__name__)

def main(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    safe_tensors: bool = False,
    delete: list[int] = [],
    prefix: str = "transformer.blocks.",
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
        train_input = input_dir / "train.safetensors"
        train_output = str(output_dir / "train.safetensors")
    else:
        model_input = input_dir / "model.pt"
        model_output = str(output_dir / "model.pt")
        train_input = input_dir / "train.pt"
        train_output = str(output_dir / "train.pt")

    logger.info("Loading config from %s", config_input)
    with open(config_input, "rt") as f:
        config = yaml.safe_load(f)

    logger.info("Loading model state from %s", model_input)
    if safe_tensors:
        model_state_dict = safetensors_file_to_state_dict(model_input)
    else:
        model_state_dict = torch.load(model_input, map_location="cpu")

    config["model"]["n_layers"] -= len(delete)
    new_model_state_dict = {}
    prev = -1
    current = -1
    for key, value in sorted(model_state_dict.items(), key=lambda x: x[0]):
        if key.startswith(prefix):
            offset = prefix.count(".")
            parts = key.split(".")
            layer = int(parts[offset])
            if layer > prev:
                current += 1
                prev = layer
            if layer in delete:
                print(f"Deleted {key} according to {delete}")
            else:
                new_key = ".".join(parts[:offset] + [str(current)] + parts[offset+1:])
                new_value = value.clone()
                new_model_state_dict[new_key] = new_value
                print(f"Cloned {key} to {new_key}")
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

    logger.info("Loading trainer state from %s", train_input)
    if safe_tensors:
        trainer_state_dict = safetensors_file_to_state_dict(train_input)
    else:
        trainer_state_dict = torch.load(train_input, map_location="cpu")
    logger.info("Saving trainer state to %s", train_output)
    if safe_tensors:
        state_dict_to_safetensors_file(trainer_state_dict, train_output)
    else:
        torch.save(trainer_state_dict, train_output)
    del trainer_state_dict

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
        "--delete",
        type=str,
        default="",
    )
    parser.add_argument(
        "--prefix",
        default="transformer.blocks.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(
        args.input_dir,
        args.output_dir,
        safe_tensors=args.safe_tensors,
        delete=[int(i) for i in args.delete.split()],
        prefix=args.prefix,
    )
