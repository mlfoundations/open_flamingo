from PIL import Image
import open_clip
import torch
from tqdm import tqdm
import torch
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def get_indices_of_unique(x):
    """
    Return the indices of x that correspond to unique elements.
    If value v is unique and two indices in x have value v, the first index is returned.
    """
    unique_elements = torch.unique(x)
    first_indices = []
    for v in unique_elements:
        indices = torch.where(x == v)[0]
        first_indices.append(indices[0])  # Take the first index for each unique element
    return torch.tensor(first_indices)


class RICES:
    def __init__(
        self, dataset, device, batch_size, world_size, rank, cached_features=None
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            cache_dir="/mmfs1/gscratch/efml/anasa2/clip_cache",
        )
        vision_encoder = vision_encoder.to(self.device)
        self.model = DDP(
            vision_encoder,
            device_ids=[self.device],
        )
        self.image_processor = image_processor

        # Precompute features
        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features

    def _precompute_features(self):
        features = []

        # Switch to evaluation mode
        self.model.eval()

        # Set up dataset for distributed eval
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset,
            shuffle=False,
            drop_last=False,
        )
        rank_indices = torch.LongTensor([i for i in sampler]).to(self.device)
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=custom_collate_fn,
        )

        with torch.no_grad():
            for batch in tqdm(
                loader, desc="Precomputing features for RICES", disable=self.rank != 0
            ):
                batch = batch["image"]
                inputs = torch.stack(
                    [self.image_processor(image) for image in batch]
                ).to(self.device)
                image_features = self.model.module.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features.detach())

        features = torch.cat(features)

        # all gather
        features_list = [torch.zeros_like(features) for _ in range(self.world_size)]
        torch.distributed.all_gather(features_list, features)
        indices_list = [torch.zeros_like(rank_indices) for _ in range(self.world_size)]
        torch.distributed.all_gather(indices_list, rank_indices)

        # concat, restore original order, and remove extra indices added by distributed sampler
        all_features, all_indices = torch.cat(features_list), torch.cat(indices_list)
        reindex = all_indices.argsort()[get_indices_of_unique(all_indices)]
        all_features = all_features[reindex]

        assert len(all_features) == len(self.dataset)
        return all_features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            inputs = torch.stack(
                [self.image_processor(image) for image in batch]
            ).to(self.device)

            # Get the feature of the input image
            query_feature = self.model.module.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]
