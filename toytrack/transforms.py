import torch
from typing import Dict

class TrackletPatchify:
    """A class to transform trackwise data into patches.

    This class splits each track into a specified number of patches, reshapes the data,
    and creates edge indices for all patch pairs.

    Attributes:
        num_patches_per_track (int): The number of patches to split each track into.

    """

    def __init__(self, num_patches_per_track: int):
        """Initialize the TrackletPatchify transform.

        Args:
            num_patches_per_track (int): The number of patches to split each track into.
        """
        self.num_patches_per_track = num_patches_per_track

    def __call__(self, sample: Dict) -> Dict:
        """Apply the TrackletPatchify transform to the input sample.

        Args:
            sample (Dict): A dictionary containing trackwise data. Must include an 'x' key
                           with a tensor of shape (num_tracks, num_hits, features).

        Returns:
            Dict: The transformed sample with patched data and additional keys:
                  - 'x': Reshaped feature tensor (num_tracks * num_patches_per_track, hits_per_patch, num_features)
                  - 'mask': Reshaped mask tensor if present in input
                  - 'pids': Repeated particle IDs for each patch if present in input
                  - 'edge_index': Tensor of patch pair indices
                  - 'y': Boolean tensor indicating if patch pairs belong to the same particle

        Raises:
            ValueError: If 'x' is not in the sample, doesn't have 3 dimensions, or if
                        num_hits is not divisible by num_patches_per_track.
        """
        if 'x' not in sample or sample['x'].dim() != 3:
            raise ValueError("Input must be trackwise data with shape (num_tracks, num_hits, features)")

        num_tracks, num_hits, num_features = sample['x'].shape
        
        if num_hits % self.num_patches_per_track != 0:
            raise ValueError(f"Number of hits ({num_hits}) must be divisible by num_patches_per_track ({self.num_patches_per_track})")

        hits_per_patch = num_hits // self.num_patches_per_track

        # Reshape x: (num_tracks * num_patches_per_track, hits_per_patch, num_features)
        sample['x'] = sample['x'].view(-1, hits_per_patch, num_features)

        # Reshape mask: (num_tracks * num_patches_per_track, hits_per_patch)
        if 'mask' in sample:
            sample['mask'] = sample['mask'].view(-1, hits_per_patch)

        # Repeat pids for each patch
        if 'pids' in sample:
            sample['pids'] = sample['pids'].repeat_interleave(self.num_patches_per_track)

        # Get N^2 patch pair combinations as edge_index
        sample['edge_index'] = torch.combinations(torch.arange(sample['x'].shape[0]), r=2)

        # Get the y truth
        sample['y'] = sample['pids'][sample['edge_index'][..., 0]] == sample['pids'][sample['edge_index'][..., 1]]

        return sample