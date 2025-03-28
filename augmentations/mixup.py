import torch
import numpy as np
from CLAPForge.core.transforms_interface import BaseWaveformTransform

class MixupAugment(BaseWaveformTransform):
    def __init__(self, alpha=1.0, p=1.0):
        """
        Initialize Mixup augmentation.
        
        Mixup combines two samples by overlaying them with a weighted combination
        and providing two sets of labels.
        
        Parameters:
            alpha (float): The alpha parameter for the beta distribution to sample the mix coefficient.
            p (float): The probability of applying this augmentation.
        """
        super().__init__(p=p)
        self.alpha = alpha

    def apply(self, input_dict, sample_rate=None):
        """
        Apply the Mixup augmentation.
        
        Parameters:
            input_dict (dict): A dictionary containing:
                - "melspecs" (torch.Tensor): Batch of mel-spectrograms with shape (batch_size, ...).
                - "labels" (torch.Tensor or np.ndarray): Corresponding labels.
            sample_rate (int): Not used here but included for interface consistency.
        
        Returns:
            dict: A dictionary with keys:
                - "melspecs": The mixed mel-spectrograms.
                - "labels": A tuple of mixed labels (first element weighted by lam, second by 1 - lam).
        """
        original_melspecs = input_dict["melspecs"]
        original_labels = input_dict["labels"]
        
        # Generate a random permutation of the batch indices.
        indices = torch.randperm(original_melspecs.size(0))
        
        # Sample the mixup coefficient from a beta distribution.
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Create the mixed mel-spectrograms.
        augmented_melspecs = original_melspecs * lam + original_melspecs[indices] * (1 - lam)
        
        # Create the corresponding mixed labels.
        augmented_labels = (original_labels * lam, original_labels[indices] * (1 - lam))
        
        return {"melspecs": augmented_melspecs, "labels": augmented_labels}
