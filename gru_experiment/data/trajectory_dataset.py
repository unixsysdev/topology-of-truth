"""
Trajectory Dataset

PyTorch Dataset for loading activation trajectories for GRU training.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TrajectoryPair:
    """A pair of student and teacher trajectories for the same question."""
    question_id: str
    student_trajectories: Dict[str, torch.Tensor]  # layer_name -> (seq, hidden)
    teacher_trajectories: Dict[str, torch.Tensor]
    student_correct: bool
    teacher_correct: bool
    student_seq_length: int
    teacher_seq_length: int


class TrajectoryDataset(Dataset):
    """
    Dataset of activation trajectory pairs.
    
    Loads pre-extracted trajectories from student and teacher models,
    paired by question for distillation training.
    """
    
    def __init__(
        self,
        data_dir: str,
        layer_name: str = "layer_9",  # Which layer to use
        max_seq_length: int = 512,
        filter_teacher_correct: bool = True,  # Only use samples where teacher was correct
        device: str = "cpu"
    ):
        self.data_dir = Path(data_dir)
        self.layer_name = layer_name
        self.max_seq_length = max_seq_length
        self.filter_teacher_correct = filter_teacher_correct
        self.device = device
        
        # Load metadata
        self.student_meta = self._load_metadata("student_metadata.jsonl")
        self.teacher_meta = self._load_metadata("teacher_metadata.jsonl")
        
        # Build paired index
        self.pairs = self._build_pairs()
        
        print(f"Loaded {len(self.pairs)} trajectory pairs")
        print(f"  Layer: {layer_name}")
        print(f"  Teacher correct filter: {filter_teacher_correct}")
        
    def _load_metadata(self, filename: str) -> Dict[str, dict]:
        """Load metadata from JSONL file."""
        meta = {}
        meta_file = self.data_dir / filename
        
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
            
        with open(meta_file) as f:
            for line in f:
                entry = json.loads(line)
                if 'question_id' in entry and 'error' not in entry:
                    meta[entry['question_id']] = entry
                    
        return meta
    
    def _build_pairs(self) -> List[str]:
        """Build list of question IDs that have both student and teacher data."""
        common_ids = set(self.student_meta.keys()) & set(self.teacher_meta.keys())
        
        pairs = []
        for qid in sorted(common_ids):
            # Optionally filter by teacher correctness
            if self.filter_teacher_correct:
                if not self.teacher_meta[qid].get('correct', False):
                    continue
            pairs.append(qid)
            
        return pairs
    
    def _load_trajectory(self, model_key: str, question_id: str) -> Dict[str, torch.Tensor]:
        """Load trajectory from npz file."""
        # Extract index from question_id (e.g., "q_0001" -> 1)
        idx = int(question_id.split('_')[1])
        
        traj_file = self.data_dir / f"{model_key}_q{idx:04d}.npz"
        
        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
            
        data = np.load(traj_file)
        trajectories = {
            k: torch.from_numpy(v) 
            for k, v in data.items()
        }
        
        return trajectories
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> TrajectoryPair:
        qid = self.pairs[idx]
        
        # Load trajectories
        student_traj = self._load_trajectory("student", qid)
        teacher_traj = self._load_trajectory("teacher", qid)
        
        # Get specific layer
        student_layer = student_traj.get(self.layer_name)
        teacher_layer = teacher_traj.get(self.layer_name)
        
        if student_layer is None or teacher_layer is None:
            raise ValueError(f"Layer {self.layer_name} not found in trajectories")
        
        # Truncate/pad to max_seq_length
        student_layer = self._truncate_pad(student_layer)
        teacher_layer = self._truncate_pad(teacher_layer)
        
        return TrajectoryPair(
            question_id=qid,
            student_trajectories={self.layer_name: student_layer},
            teacher_trajectories={self.layer_name: teacher_layer},
            student_correct=self.student_meta[qid].get('correct', False),
            teacher_correct=self.teacher_meta[qid].get('correct', False),
            student_seq_length=self.student_meta[qid].get('seq_length', 0),
            teacher_seq_length=self.teacher_meta[qid].get('seq_length', 0)
        )
    
    def _truncate_pad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Truncate or pad sequence to max_seq_length."""
        seq_len = tensor.shape[0]
        
        if seq_len > self.max_seq_length:
            # Truncate (take last max_seq_length tokens)
            return tensor[-self.max_seq_length:]
        elif seq_len < self.max_seq_length:
            # Pad (add zeros at the beginning)
            padding = torch.zeros(
                self.max_seq_length - seq_len, 
                *tensor.shape[1:],
                dtype=tensor.dtype
            )
            return torch.cat([padding, tensor], dim=0)
        else:
            return tensor


def collate_trajectories(batch: List[TrajectoryPair]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    
    # Stack trajectories
    layer_name = list(batch[0].student_trajectories.keys())[0]
    
    student_traj = torch.stack([
        item.student_trajectories[layer_name] 
        for item in batch
    ])  # (batch, seq, hidden)
    
    teacher_traj = torch.stack([
        item.teacher_trajectories[layer_name]
        for item in batch
    ])  # (batch, seq, hidden)
    
    # Collect metadata
    student_correct = torch.tensor([item.student_correct for item in batch])
    teacher_correct = torch.tensor([item.teacher_correct for item in batch])
    
    return {
        'student_trajectories': student_traj,
        'teacher_trajectories': teacher_traj,
        'student_correct': student_correct,
        'teacher_correct': teacher_correct,
        'question_ids': [item.question_id for item in batch]
    }


def create_dataloader(
    data_dir: str,
    layer_name: str = "layer_9",
    batch_size: int = 4,
    max_seq_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    filter_teacher_correct: bool = True
) -> DataLoader:
    """Create DataLoader for trajectory dataset."""
    
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        layer_name=layer_name,
        max_seq_length=max_seq_length,
        filter_teacher_correct=filter_teacher_correct
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_trajectories
    )


# Test
if __name__ == "__main__":
    import sys
    
    # Check if data exists
    data_dir = Path("gru_experiment/data/trajectories")
    
    if not data_dir.exists():
        print("Data directory doesn't exist. Run prepare_trajectories.py first.")
        sys.exit(1)
        
    # Try to create dataset
    try:
        dataset = TrajectoryDataset(
            data_dir=str(data_dir),
            layer_name="layer_9",
            max_seq_length=256
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            item = dataset[0]
            print(f"Sample item:")
            print(f"  Question ID: {item.question_id}")
            print(f"  Student correct: {item.student_correct}")
            print(f"  Teacher correct: {item.teacher_correct}")
            for k, v in item.student_trajectories.items():
                print(f"  {k} shape: {v.shape}")
                
    except Exception as e:
        print(f"Error: {e}")
