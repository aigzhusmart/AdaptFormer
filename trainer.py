import logging
import os

import torch
import transformers
from transformers import HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from src.adaptformer.configuration_adaptformer import AdaptFormerConfig
from src.adaptformer.modeling_adaptformer import AdaptFormerForChangeDetection
from src.args.data_args import DataArguments
from src.loader import CDDataset
from src.metrics import ConfuseMatrixMeter

logger = logging.getLogger(__name__)


def train(training_args: TrainingArguments, data_args: DataArguments):
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    model = AdaptFormerForChangeDetection(AdaptFormerConfig())

    training_set = CDDataset(
        root_dir=data_args.root_dir,
        split="train",
        img_size=data_args.image_size,
        is_train=True,
        label_transform="norm",
    )
    val_set = CDDataset(
        root_dir=data_args.root_dir,
        split="test",
        img_size=data_args.image_size,
        is_train=False,
        label_transform="norm",
    )

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = ConfuseMatrixMeter(2)
            metrics.update_cm(pr=pred_labels, gt=labels)
            scores = metrics.get_scores()
        return scores

    trainer = Trainer(
        model,
        training_args,
        compute_metrics=compute_metrics,
        train_dataset=training_set,
        eval_dataset=val_set,
    )
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metirc = trainer.evaluate()
    logger.info(train_result)
    logger.info(metirc)
    model.save_pretrained(f"{training_args.output_dir}/best")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments))
    args = parser.parse_json_file("training_config.json")
    training_args: TrainingArguments = args[0]
    data_args: DataArguments = args[1]
    train(training_args, data_args)
