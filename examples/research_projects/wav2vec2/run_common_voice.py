#!/usr/bin/env python3
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import unidecode
import soundfile
import datasets
import librosa
import numpy as np
import torch
import torchaudio
from packaging import version
from torch import nn

from audiomentations import Compose, AddGaussianNoise, Gain, PitchShift, Shift

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.trainer_pt_utils import LengthGroupedSampler


if is_apex_available():
    from apex import amp


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    load_processor: Optional[bool] = field(
        default=True,
        metadata={"help": "Should the processor/tokenizer be loaded with the checkpointed model or set up a new LM head"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: Optional[str] = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    chars_to_ignore: List[str] = list_field(
        default=[r'!"#$%&()*+,./:;<=>?@\[\]\\_{}|~£¤¨©ª«¬®¯°·¸»¼½¾ðʺ˜˝ˮ‐–—―‚“”„‟•…″‽₋€™−√�'],
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    augmented: bool = field(
        default=False,
        metadata={
            "help": "Duplicate training data, with acoustic manipulations"
        }
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    
# solution from https://discuss.huggingface.co/t/spanish-asr-fine-tuning-wav2vec2/4586/6
class GroupedLengthsTrainer(CTCTrainer):
    # length_field_name should possibly be part of TrainingArguments instead
    def __init__(self, train_seq_lengths:List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_seq_lengths = train_seq_lengths
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            # lengths = self.train_dataset[self.length_field_name] if self.length_field_name is not None else None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.train_dataset, self.args.train_batch_size, lengths=self.train_seq_lengths, model_input_name=model_input_name
            )
        else:
            return super()._get_train_sampler()
    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
        
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        logger.info(f"Looking for last checkpoint in {training_args.output_dir}")
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )



    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets:
    logger.info("Loading the raw audio files")
    train_dataset = datasets.load_dataset(
        "common_voice", data_args.dataset_config_name, split=data_args.train_split_name, cache_dir=model_args.cache_dir
    )
    # train_dataset = train_dataset.filter(lambda example: example["up_votes"] > example["down_votes"])
    eval_dataset = datasets.load_dataset("common_voice", data_args.dataset_config_name, split=data_args.eval_split_name, cache_dir=model_args.cache_dir)

    logger.info("Normalizing transcriptions")
    chars_to_ignore_regex = f'[{"".join(data_args.chars_to_ignore)}]'

    def remove_special_characters(batch):
        batch["text"] = re.sub(r'[‘’´`]', r"'", batch["sentence"])
        batch["text"] = re.sub(chars_to_ignore_regex, "", batch["text"]).lower().strip() + " "
        batch["text"] = re.sub(r"(-|' | '|  +)", " ", batch["text"])
        batch["text"] = unidecode.unidecode(batch["text"])
        return batch

    train_dataset = train_dataset.map(remove_special_characters, remove_columns=["sentence"], num_proc=data_args.preprocessing_num_workers)
    train_dataset = train_dataset.filter(lambda example: example["down_votes"] == 0)
    eval_dataset = eval_dataset.map(remove_special_characters, remove_columns=["sentence"], num_proc=data_args.preprocessing_num_workers)
    eval_dataset = train_dataset.filter(lambda example: example["down_votes"] == 0)

    def extract_all_chars(batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if last_checkpoint and model_args.load_processor:
        logger.info(f"Loading pretrained processor from {model_args.model_name_or_path}")
        processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)
        logger.info(processor.tokenizer.get_vocab())
    else:
        logger.info("Creating tokenizer and processor")
        vocab_train = train_dataset.map(
            extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=train_dataset.column_names,
        )
        vocab_test = eval_dataset.map(
            extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=eval_dataset.column_names,
        )

        vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        with open("vocab.json", "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)


        tokenizer = Wav2Vec2CTCTokenizer(
            "vocab.json",
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(training_args.output_dir)
        
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        gradient_checkpointing=model_args.gradient_checkpointing,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        # vocab_size=len(processor.tokenizer),
    )

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    logger.info("Vectorizing audio files")
    resampler = torchaudio.transforms.Resample(48_000, 16_000)

    augment = Compose([
        AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.01, p=1),
        PitchShift(min_semitones=-3, max_semitones=3, p=0.8),
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.8),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.8),
    ])
    
    # Preprocessing the datasets.
    # We need to read the aduio files as arrays and tokenize the targets.
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        batch["sampling_rate"] = 16_000
        batch["target_text"] = batch["text"]
        return batch

    def augmented_speech_file_to_array_fn(batch):
        try:
            speech_array, sampling_rate = soundfile.read(batch["path"] + "-augmented.wav")
        except:
            speech_array, sampling_rate = torchaudio.load(batch["path"])
            speech_array = resampler(speech_array)
            speech_array = augment(samples=speech_array, sample_rate=sampling_rate).squeeze()
            soundfile.write(batch["path"]+"-augmented.wav", speech_array, sampling_rate, subtype='PCM_24')

        batch["speech"] = speech_array
        batch["sampling_rate"] = 16_000
        batch["target_text"] = batch["text"]
        return batch
    
    if data_args.augmented:
        logger.info("Creating augmented audio data")
        train_augmented = train_dataset.map(
            augmented_speech_file_to_array_fn, 
            remove_columns=train_dataset.column_names, 
            num_proc=data_args.preprocessing_num_workers
        )
        
    train_dataset = train_dataset.map(
        speech_file_to_array_fn,
        remove_columns=train_dataset.column_names,
        num_proc=1,
    )
    eval_dataset = eval_dataset.map(
        speech_file_to_array_fn,
        remove_columns=eval_dataset.column_names,
        num_proc=1,
    )

    if data_args.augmented:
        train_dataset = datasets.concatenate_datasets([train_dataset, train_augmented])
        
    # remove data that is too long to avoid getting GPU out of memory error
    train_dataset = train_dataset.filter(lambda batch: len(batch["speech"]) < 150000)
        
    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        # Setup the processor for targets
        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )
    logging.info(f"Training set is {len(train_dataset)} examples.")
    eval_dataset = eval_dataset.map(
        prepare_dataset,
        remove_columns=eval_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )
    logging.info(f"Validation set is {len(eval_dataset)} examples.")

    # Metric
    wer_metric = datasets.load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize our Trainer
    logger.info("Initializing the trainer")
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        logger.info("Beginning training")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results


if __name__ == "__main__":
    main()
