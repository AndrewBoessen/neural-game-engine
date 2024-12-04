import numpy as np
import torch
from scipy import linalg
from transformers import VideoMAEFeatureExtractor, VideoMAEModel


class GameEngineValidationMetrics:
    @staticmethod
    def calculate_fvd(real_videos, generated_videos, feature_extractor=None):
        """
        Calculate Frechet Video Distance (FVD) between real and generated videos.

        Args:
            real_videos (torch.Tensor): Tensor of real videos
                Shape: (num_real_videos, frames, channels, height, width)
            generated_videos (torch.Tensor): Tensor of generated videos
                Shape: (num_generated_videos, frames, channels, height, width)
            feature_extractor (nn.Module, optional): Pre-trained feature extraction network

        Returns:
            float: Frechet Video Distance score
        """
        # Use VideoMAE as default feature extractor if none provided
        if feature_extractor is None:
            feature_extractor = VideoGPTFeatureExtractor()

        # Extract features
        real_features = feature_extractor.extract_features(real_videos)
        generated_features = feature_extractor.extract_features(generated_videos)

        # Calculate mean and covariance
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(
            real_features, rowvar=False
        )
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(
            generated_features, rowvar=False
        )

        # Calculate FVD
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    @staticmethod
    def calculate_psnr(real_videos, generated_videos, max_pixel_value=1.0):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) for video sequences.

        Args:
            real_videos (torch.Tensor): Tensor of real videos
                Shape: (num_videos, frames, channels, height, width)
            generated_videos (torch.Tensor): Tensor of generated videos
                Shape: (num_videos, frames, channels, height, width)
            max_pixel_value (float): Maximum possible pixel value

        Returns:
            float: Average PSNR across all videos and frames
        """
        # Ensure videos are of the same shape
        assert (
            real_videos.shape == generated_videos.shape
        ), "Video tensors must have identical shapes"

        # Flatten videos for easier computation
        real_videos_flat = real_videos.view(-1)
        generated_videos_flat = generated_videos.view(-1)

        # Calculate Mean Squared Error (MSE)
        mse = torch.mean((real_videos_flat - generated_videos_flat) ** 2)

        # Calculate PSNR
        if mse == 0:
            return float("inf")

        psnr = 20 * torch.log10(torch.tensor(max_pixel_value)) - 10 * torch.log10(mse)
        return float(psnr.item())

    @staticmethod
    def validate_game_engine(
        real_videos,
        generated_videos,
        fvd_weight=0.5,
        psnr_weight=0.5,
        psnr_threshold=30.0,
    ):
        """
        Comprehensive validation of game engine video generation.

        Args:
            real_videos (torch.Tensor): Ground truth videos
            generated_videos (torch.Tensor): Generated videos
            fvd_weight (float): Weight for FVD in final score
            psnr_weight (float): Weight for PSNR in final score
            psnr_threshold (float): Minimum acceptable PSNR

        Returns:
            dict: Validation metrics and overall score
        """
        # Calculate individual metrics
        fvd_score = GameEngineValidationMetrics.calculate_fvd(
            real_videos, generated_videos
        )
        psnr_score = GameEngineValidationMetrics.calculate_psnr(
            real_videos, generated_videos
        )

        # Normalize and weight the scores
        # Lower FVD is better, so we invert its scale
        normalized_fvd = 1 / (1 + fvd_score)
        normalized_psnr = psnr_score / psnr_threshold

        # Compute weighted final score
        final_score = (fvd_weight * normalized_fvd + psnr_weight * normalized_psnr) / (
            fvd_weight + psnr_weight
        )

        return {
            "fvd_score": fvd_score,
            "psnr_score": psnr_score,
            "normalized_fvd": normalized_fvd,
            "normalized_psnr": normalized_psnr,
            "final_validation_score": final_score,
        }


class VideoGPTFeatureExtractor:
    def __init__(self, model_name="MCG-NJU/videomae-base-finetuned-kinetics"):
        """
        Initialize pre-trained VideoMAE model for feature extraction

        Args:
            model_name (str): Hugging Face model identifier
        """
        self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, videos):
        """
        Extract features from video tensors

        Args:
            videos (torch.Tensor): Video tensor
                Shape: (batch, frames, channels, height, width)

        Returns:
            np.ndarray: Extracted feature vectors
        """
        # Preprocess videos
        inputs = self.feature_extractor(
            list(videos.permute(0, 2, 1, 3, 4).numpy()), return_tensors="pt"
        )

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # Pool across sequence

        return features.numpy()
