# training.py
import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
from datasets import load_dataset
from model import CustomDeepSeekV3
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data_stream(dataset_name, split, tokenizer, block_size, config=None):
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=block_size)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding if available
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a new [PAD] token
            print(f"Added new padding token: {tokenizer.pad_token}")

    dataset = load_dataset(dataset_name, name=config, split=split, streaming=True)
    dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
    return dataset


def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in batch])
    return input_ids


def generate_tokens(model, tokenizer, prompt, max_length=50, device="cuda"):
    """Generates output tokens based on a given prompt."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = input_ids
        for _ in range(max_length):
            logits = model(outputs[:, -1:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            outputs = torch.cat([outputs, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Save the model, optimizer, scheduler, and training state
def save_checkpoint(config, model, optimizer, scheduler, step, loss):
    checkpoint_path = os.path.join(config['checkpoints']['checkpoints_path'], f"checkpoint_{step}.pth")
    checkpoint = {
        'model_state_dict': model.state_dict(),  # Save model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,  # Save scheduler state (optional)
        'step': step,  # Save current training step
        'loss': loss,  # Save the most recent loss
        'config': config  # Save training configuration for reference
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


# Load checkpoint and resume training
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state

    step = checkpoint['step']  # Resume training step
    loss = checkpoint.get('loss', None)  # Get last loss (if saved)
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from step {step}")

    return step, loss


def update_routing_bias(model, input_ids):
    for i in range(len(model.layers)):
        expert_load = torch.zeros(model.layers[i].feed_forward.num_routed_experts, device=device)
        for k in range(model.layers[i].feed_forward.top_k_experts):
            routing_logits = model.layers[i].feed_forward.router(input_ids) + model.layers[i].feed_forward.routing_bias
            routing_probs = torch.sigmoid(routing_logits)
            _, indices = torch.topk(routing_probs, model.layers[i].feed_forward.top_k_experts, dim=-1)

            for idx in range(model.layers[i].feed_forward.num_routed_experts):
                expert_load[idx] += (indices[..., k] == idx).sum()

        expert_load = expert_load / (input_ids.size(0) * input_ids.size(1) * model.layers[i].feed_forward.top_k_experts)
        model.layers[i].feed_forward.update_bias_terms(expert_load)


def train(config):

    ## Speed up with malmul
    torch.set_float32_matmul_precision('high')

    # Load model and tokenizer
    model = CustomDeepSeekV3(config['model']['model_config'])

    summary(
        model,
        input_size=(1, config['model']['model_config']['sequence_length']),  # Example input size: (batch_size=1, seq_length=128)
        dtypes=[torch.long],  # Specify input data type
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,  # Adjust column width for better readability
        depth=3  # Adjust depth to show nested layers
    )

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['tokenizer_name_or_path'])


    # Load data with streaming
    config_name = "cosmopedia-v2"

    # Load the dataset with the specified config
    train_dataset = load_data_stream(
        dataset_name=config['data_stages']['data']['dataset_name'],
        split="train",
        tokenizer=tokenizer,  # Ensure tokenizer is defined
        block_size=config['model']['model_config']['sequence_length'],  # Adjust block size as needed
        config=config_name  # Pass the configuration explicitly
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['tokens']['batch_size'],
        collate_fn=collate_fn
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['learning_rate_scheduler']['learning_rate'],
        betas=(
            config['optimizer']['optimizer_factory']['adam_beta1'],
            config['optimizer']['optimizer_factory']['adam_beta2']
        ),
        eps=config['optimizer']['optimizer_factory']['adam_eps'],
        weight_decay=config['optimizer']['weight_decay']
    )

    # Training loop
    model.train()
    model.to(device)

    ### Torch Compile applied. Comment it on mac and windows
    #model = torch.compile(model)

    # Load checkpoint if available
    resume_checkpoint_path = config['checkpoints']['resume_checkpoint_path']
    start_step = 0
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        start_step, loss = load_checkpoint(resume_checkpoint_path, model, optimizer)

    max_steps = config['tokens']['train_steps']
    sample_prompt = "United States of America and India both have one common shared principal and that is"

    for step, batch in enumerate(train_dataloader, start=start_step):
        if step >= max_steps:
            print("Reached maximum training steps.")
            save_checkpoint(config, model, optimizer, None, step, loss)
            generated_text = generate_tokens(model, tokenizer, sample_prompt, max_length=50, device=device)
            print(f"Validation: (Step {step}), Generated text: {generated_text}")
            break

        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        #### AutoCast
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(batch)
            # Shift the targets (labels) by 1
            shifted_logits = outputs[:, :-1, :].contiguous()  # Remove the last token from outputs
            shifted_labels = batch[:, 1:].contiguous()  # Remove the first token from labels

            # Flatten the tensors for loss computation
            logits_flat = shifted_logits.view(-1, config['model']['model_config']['vocab_size'])
            labels_flat = shifted_labels.view(-1)

            # Compute the cross-entropy loss
            loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad'])
        optimizer.step()

        update_routing_bias(model, batch)


        if step % config['logging']['iteration_step_info_interval'] == 0:
            print(f"Step {step}, Loss: {loss.item()}")

        if step % config['checkpoints']['checkpoint_interval'] == 0:
            #checkpoint_path = os.path.join(config['checkpoints']['checkpoints_path'], f"checkpoint_{step}.pth")
            save_checkpoint(config, model, optimizer, None, step, loss)

        if step % config['tokens']['val_check_interval'] == 0:
            generated_text = generate_tokens(model, tokenizer, sample_prompt, max_length=50, device=device)
            print(f"Validation: (Step {step}), Generated text: {generated_text}")

    print("Training complete.")


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
