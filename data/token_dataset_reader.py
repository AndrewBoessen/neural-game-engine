import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    def __init__(
        self, image_tokens_path, action_tokens_path, context_length=2048, stride=1024
    ):
        """
        Dataset for handling large context length sequences.

        Args:
            image_tokens_path (str): Path to .npy file with image tokens
            action_tokens_path (str): Path to .npy file with action tokens
            context_length (int): Maximum context length for model input
            stride (int): Sliding window stride between chunks
        """
        # Load original data
        self.image_tokens = np.load(image_tokens_path)
        self.action_tokens = np.load(action_tokens_path)

        # Validate data
        assert len(self.image_tokens) == len(
            self.action_tokens
        ), "Image tokens and action tokens must have the same length"

        # Prepare chunked sequences
        self.chunks = self._create_chunks(context_length, stride)

    def _create_chunks(self, context_length, stride):
        """
        Create chunks of sequences with sliding window.

        Args:
            context_length (int): Length of each chunk
            stride (int): Sliding window stride

        Returns:
            list: List of tuples (image_chunk, action_chunk)
        """
        chunks = []
        # Create sliding window chunks
        for start in range(0, len(self.image_tokens) - context_length + 1, stride):
            end = start + context_length
            image_chunk = self.image_tokens[start:end]
            action_chunk = self.action_tokens[start:end]
            chunks.append((image_chunk, action_chunk))

        return chunks

    def __len__(self):
        """
        Returns total number of chunks.

        Returns:
            int: Number of sequence chunks
        """
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        Retrieve a specific chunk.

        Args:
            idx (int): Index of the chunk

        Returns:
            tuple: (image_tokens, action_tokens, original_sequence_index)
        """
        image_chunk, action_chunk = self.chunks[idx]

        # Convert to torch tensors
        image_tokens = torch.from_numpy(image_chunk).long()
        action_tokens = torch.from_numpy(action_chunk).long()

        return image_tokens, action_tokens

    def get_original_sequence(self, orig_idx):
        """
        Retrieve the original full sequence for a given index.

        Args:
            orig_idx (int): Original sequence index

        Returns:
            tuple: (full_image_tokens, full_action_tokens)
        """
        return (
            torch.from_numpy(self.image_tokens[orig_idx]).long(),
            torch.from_numpy(self.action_tokens[orig_idx]).long(),
        )
