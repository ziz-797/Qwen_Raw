import transformers
from dataclasses import dataclass
from typing import Dict, Sequence, List, Any
import torch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        # 1. Prepare text inputs for padding
        input_ids = [instance["input_ids"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # Determine padding side
        padding_side = instances[0].get("padding_side", self.tokenizer.padding_side)
        self.tokenizer.padding_side = padding_side
        
        # Use tokenizer to pad input_ids and attention_mask
        features = [{"input_ids": input_ids[i], "attention_mask": attention_mask[i]} for i in range(len(instances))]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        
        # 2. Pad labels manually to ensure -100 padding
        if labels[0] is not None:
            if padding_side == "right":
                batch["labels"] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            else:
                # Left padding: reverse, pad, reverse
                reversed_labels = [label.flip(0) for label in labels]
                padded_reversed = torch.nn.utils.rnn.pad_sequence(reversed_labels, batch_first=True, padding_value=-100)
                batch["labels"] = padded_reversed.flip(1)
        else:
            batch["labels"] = None
            
        # 3. Handle images (concatenate instead of stack)
        if "pixel_values" in instances[0]:
            pv = [ins["pixel_values"] for ins in instances]
            pv = [x.squeeze(0) if x.dim() == 4 and x.size(0) == 1 else x for x in pv]
            batch["pixel_values"] = torch.stack(pv, dim=0) 
        # 4. Pass through other metadata
        for key in ["findings", "prompt", "id", "image_size"]:
            if key in instances[0]:
                batch[key] = [instance[key] for instance in instances]
                
        return batch