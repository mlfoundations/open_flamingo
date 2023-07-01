from PIL import Image
import open_clip
import torch
from tqdm import tqdm
import numpy as np

class RICES:
    def __init__(self, dataset, device, batch_size):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", cache_dir="/mmfs1/gscratch/efml/anasa2/clip_cache"
        )
        self.model = vision_encoder.to(self.device)
        self.image_processor = image_processor

        # Precompute features
        self.features = self._precompute_features()

    def _precompute_features(self):
        features = []

        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(self.dataset), self.batch_size), desc="Precomputing features for RICES"):
                batch = [self.dataset[j]["image"] for j in range(i, min(i + self.batch_size, len(self.dataset)))]
                inputs = torch.stack([self.image_processor(image) for image in batch]).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features.detach().cpu())

        return torch.cat(features)

    def find(self, pillow_image, num_examples):
        # Transform the input image
        image_input = self.image_processor(pillow_image).unsqueeze(0).to(self.device)

        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Get the feature of the input image
            query_feature = self.model.encode_image(image_input)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            # Compute the similarity of the input image to the precomputed features
            similarity = (self.features @ query_feature.T).squeeze(1)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(descending=True)[:num_examples]

            return [self.dataset[i] for i in indices.tolist()]
