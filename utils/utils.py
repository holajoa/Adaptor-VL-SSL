import timm

import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
import os


def get_image_embeds_raw(
    dataloader,
    vision_model,
    vision_model_type="huggingface",
    save_path="./weights/image_embeds",
    batch_size=32,
    embedding_dim=768,
    model_name="",
    split="train",
    device="cuda",
    full=False,
):
    if not model_name:
        model_name = vision_model_type
    model_name = model_name.replace("/", "_")

    if vision_model_type in ["timm", "ae", "hub"]:
        device = next(
            vision_model.parameters()
        ).device  # timm-loaded model is a nn.Sequential
    else:
        device = vision_model.device

    if full:
        vision_model.eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(tqdm(dataloader)):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device=device)
                if batch_idx == 0:
                    assert (
                        inputs["pixel_values"].size(0) == batch_size
                    ), f"Expected batch size {batch_size}, got {inputs.pixel_values.size(0)}"

                if vision_model_type == "huggingface":
                    vision_outputs = vision_model(
                        inputs.pixel_values,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    image_embeds_raw = vision_outputs.last_hidden_states
                elif vision_model_type == "timm":
                    image_embeds_raw = vision_model(inputs["pixel_values"])  # [:, 0, :]
                elif vision_model_type == "hub":
                    output = vision_model(inputs["pixel_values"], is_training=True)
                    local = output["x_norm_patchtokens"]
                    cls = output["x_norm_clstoken"].unsqueeze(1)
                    image_embeds_raw = torch.concat([cls, local], dim=1)
                elif vision_model_type == "ae":
                    vision_outputs = vision_model.decode(inputs["pixel_values"])
                    image_embeds_raw = torch.flatten(
                        vision_outputs, start_dim=2
                    ).permute(
                        (0, 2, 1)
                    )  # .mean(1)
                else:
                    raise ValueError(f"{vision_model_type} is not supported.")
                assert (
                    len(image_embeds_raw.size()) == 3
                ), f"Expected 3D tensor, got {image_embeds_raw.size()}"

                if batch_idx == 0:
                    seq_len = image_embeds_raw.shape[1]
                    out = np.zeros(
                        (len(dataloader.dataset), seq_len, embedding_dim),
                        dtype=np.float32,
                    )

                if batch_idx == num_batches - 1:
                    out[batch_idx * batch_size :] = (
                        image_embeds_raw.detach().cpu().numpy()
                    )
                    break
                out[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                    image_embeds_raw.detach().cpu().numpy()
                )

            np.save(os.path.join(save_path, f"{model_name}_{split}.npy"), out)
    else:
        # Initialise empty npy
        num_batches = len(dataloader)
        out = np.zeros((len(dataloader.dataset), embedding_dim), dtype=np.float32)

        vision_model.eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(tqdm(dataloader)):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device=device)
                if batch_idx == 0:
                    assert (
                        inputs["pixel_values"].size(0) == batch_size
                    ), f"Expected batch size {batch_size}, got {inputs.pixel_values.size(0)}"

                if vision_model_type == "huggingface":
                    vision_outputs = vision_model(
                        inputs.pixel_values,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    image_embeds_raw = vision_outputs.pooler_output
                elif vision_model_type == "timm":
                    image_embeds_raw = vision_model(inputs["pixel_values"])[:, 0, :]
                elif vision_model_type == "hub":
                    image_embeds_raw = vision_model(inputs["pixel_values"])
                elif vision_model_type == "ae":
                    vision_outputs = vision_model.decode(inputs["pixel_values"])
                    image_embeds_raw = (
                        torch.flatten(vision_outputs, start_dim=2)
                        .permute((0, 2, 1))
                        .mean(1)
                    )
                else:
                    raise ValueError(f"{vision_model_type} is not supported.")
                assert (
                    len(image_embeds_raw.size()) == 2
                ), f"Expected 2D tensor, got {image_embeds_raw.size()}"

                if batch_idx == num_batches - 1:
                    out[batch_idx * batch_size :] = (
                        image_embeds_raw.detach().cpu().numpy()
                    )
                    break
                out[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                    image_embeds_raw.detach().cpu().numpy()
                )

            np.save(os.path.join(save_path, f"{model_name}_{split}.npy"), out)


def get_text_embeds_raw(
    dataloader,
    text_model,
    save_path="./weights/text_embeds",
    removed_arguments=["cap_lens", "pixel_values", "path", "return_loss"],
    batch_size=32,
    embedding_dim=768,
    model_name="",
    split="train",
    device="cuda",
):
    assert model_name, "model_name must be specified."
    model_name = model_name.replace("/", "_")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialise empty npy
    num_batches = len(dataloader)
    out = np.zeros((len(dataloader.dataset), embedding_dim), dtype=np.float32)

    # device = text_model.device
    text_model.eval()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(dataloader)):
            [inputs.pop(key, None) for key in removed_arguments]
            for k, v in inputs.items():
                inputs[k] = v.to(device=device)
            text_embeds_raw = text_model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            ).pooler_output
            if batch_idx == num_batches - 1:
                out[batch_idx * batch_size :] = text_embeds_raw.detach().cpu().numpy()
                break
            out[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                text_embeds_raw.detach().cpu().numpy()
            )
        np.save(os.path.join(save_path, f"{model_name}_{split}.npy"), out)


def set_environment_for_aml(num_nodes, gpus_per_node):
    if num_nodes > 1:
        os.environ["MASTER_ADDRESS"], os.environ["MASTER_PORT"] = os.environ.get(
            "AZ_BATCH_MASTER_NODE"
        ).split(":")

    else:
        os.environ["MASTER_ADDRESS"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = "47586"

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDRESS")

    os.environ["NODE_RANK"] = str(
        int(os.environ.get("OMPI_COMM_WORLD_RANK")) // gpus_per_node
    )
    os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "")
    os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE")
