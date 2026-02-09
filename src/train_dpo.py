import inspect
import traceback
import torch
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig, __version__ as trl_version

from src.config import *
from src.model import load_model_and_tokenizer
from data.prepare_dataset import load_mepo_dataset

def build_dpo_trainer_safely(model, tokenizer, training_args, dataset, dpo_config):
    """
    Build a DPOTrainer in a version-robust way by inspecting DPOTrainer.__init__.
    Returns an instantiated trainer.
    """
    print("trl version:", trl_version)
    init_sig = inspect.signature(DPOTrainer.__init__)
    param_names = [p for p in init_sig.parameters.keys() if p != "self"]
    print("DPOTrainer.__init__ parameters:", param_names)

    # Known mapping of common parameter names -> runtime objects
    known_args = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": dataset,
        "tokenizer": tokenizer,
        "dpo_config": dpo_config,
        "config": dpo_config,
        "beta": BETA,
        "max_length": MAX_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        # legacy names sometimes used
        "dataset": dataset,
        "train_dataset": dataset,
    }

    # Build kwargs only for accepted parameters
    kwargs = {}
    for name in param_names:
        if name in known_args:
            kwargs[name] = known_args[name]

    print("Calling DPOTrainer with kwargs:", list(kwargs.keys()))

    # Try keyword call first
    try:
        trainer = DPOTrainer(**kwargs)
        return trainer
    except TypeError as e:
        print("DPOTrainer(**kwargs) failed:", e)
        print("Falling back to some positional attempts...")

    # Fallback: try a few plausible positional orders.
    # These cover older variants that expect positional arguments: (model, ref_model, args, train_dataset, tokenizer)
    try:
        trainer = DPOTrainer(model, None, training_args, dataset, tokenizer)
        return trainer
    except Exception as e:
        print("Positional fallback failed:", e)
        traceback.print_exc()

    # Final fallback: attempt keyword calls with minimal args
    minimal_kwargs = {}
    for k in ("model", "args", "train_dataset", "tokenizer"):
        if k in known_args and k in param_names:
            minimal_kwargs[k] = known_args[k]
    print("Trying minimal kwargs:", minimal_kwargs.keys())
    try:
        trainer = DPOTrainer(**minimal_kwargs)
        return trainer
    except Exception as e:
        print("Minimal kwargs failed:", e)
        traceback.print_exc()

    raise RuntimeError("Could not construct DPOTrainer with detected signature. See printed tracebacks above.")

def train():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    model, tokenizer = load_model_and_tokenizer()
    dataset = load_mepo_dataset(limit=5000)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # ----------------------------------------------------------------
    # Compatibility shim for TRL versions that expect extra fields
    #  Add safe defaults so DPOTrainer won't crash.
    # ----------------------------------------------------------------
    # TRL checks attributes on args; ensure they exist with safe values.
    if not hasattr(training_args, "model_init_kwargs"):
        training_args.model_init_kwargs = None
    if not hasattr(training_args, "ref_model_init_kwargs"):
        training_args.ref_model_init_kwargs = None
    if not hasattr(training_args, "push_to_hub"):
        training_args.push_to_hub = False
    if not hasattr(training_args, "run_name"):
        training_args.run_name = None
    if not hasattr(training_args, "pad_token"):
        training_args.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if not hasattr(training_args, "generate_during_eval"):
        training_args.generate_during_eval = False
    if not hasattr(training_args, "model_adapter_name"):
        training_args.model_adapter_name = None
    if not hasattr(training_args, "ref_adapter_name"):
        training_args.ref_adapter_name = None
    if not hasattr(training_args, "reference_free"):
        training_args.reference_free = False
    if not hasattr(training_args, "disable_dropout"):
        training_args.disable_dropout = False
    if not hasattr(training_args, "label_pad_token_id"):
        training_args.label_pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    if not hasattr(training_args, "max_prompt_length"):
        training_args.max_prompt_length = MAX_PROMPT_LENGTH
    if not hasattr(training_args, "max_completion_length"):
        training_args.max_completion_length = MAX_COMPLETION_LENGTH
    if not hasattr(training_args, "max_length"):
        training_args.max_length = MAX_LENGTH
    if not hasattr(training_args, "truncation_mode"):
        training_args.truncation_mode = "round_robin"
    if not hasattr(training_args, "precompute_ref_log_probs"):
        training_args.precompute_ref_log_probs = False
    if not hasattr(training_args, "use_logits_to_keep"):
        training_args.use_logits_to_keep = False
    if not hasattr(training_args, "padding_free"):
        training_args.padding_free = False
    if not hasattr(training_args, "beta"):
        training_args.beta = BETA
    if not hasattr(training_args, "label_smoothing"):
        training_args.label_smoothing = 0.0
    if not hasattr(training_args, "loss_type"):
        training_args.loss_type = "sigmoid"
    if not hasattr(training_args, "loss_weights"):
        training_args.loss_weights = None
    if not hasattr(training_args, "use_weighting"):
        training_args.use_weighting = False
    if not hasattr(training_args, "f_divergence_type"):
        training_args.f_divergence_type = "kl"
    if not hasattr(training_args, "f_alpha_divergence_coef"):
        training_args.f_alpha_divergence_coef = 1.0
    if not hasattr(training_args, "dataset_num_proc"):
        training_args.dataset_num_proc = 1
    if not hasattr(training_args, "tools"):
        training_args.tools = []
    if not hasattr(training_args, "sync_ref_model"):
        training_args.sync_ref_model = False
    if not hasattr(training_args, "rpo_alpha"):
        training_args.rpo_alpha = 0.0
    if not hasattr(training_args, "ld_alpha"):
        training_args.ld_alpha = 0.0





    # DPOConfig (still useful where available)
    try:
        dpo_config = DPOConfig(
            beta=BETA,
            max_length=MAX_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH
        )
    except Exception as e:
        # If DPOConfig isn't available or has different signature, set to None and rely on beta kwargs
        print("Could not create DPOConfig object, will pass beta/max_length if supported. Error:", e)
        dpo_config = None

    # Build trainer in a version-robust way
    trainer = build_dpo_trainer_safely(model, tokenizer, training_args, dataset, dpo_config)

    print("Trainer constructed:", type(trainer), "Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Training complete - saved to", OUTPUT_DIR)

if __name__ == "__main__":
    train()
