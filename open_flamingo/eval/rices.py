import open_clip
import torch
from tqdm import tqdm
import torch
from utils import custom_collate_fn


class RICES:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        vision_encoder_path="ViT-B-32",
        vision_encoder_pretrained="openai",
        cached_features=None,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
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

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                batch = batch["image"]
                inputs = torch.stack(
                    [self.image_processor(image) for image in batch]
                ).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features.detach())

        features = torch.cat(features)
        return features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]
