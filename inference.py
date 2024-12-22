import os
import json

from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet

from utils.decoding_utils import greedy_decoding
from decoder import CaptionDecoder



class ImageCaptioningTokenizer:
    def __init__(self, config):
        # Load the vocabulary mappings
        with open(config["word2idx_path"], "r", encoding="utf8") as f:
            self._word2idx = json.load(f)
        self._idx2word = {str(idx): word for word, idx in self._word2idx.items()}

        # Auxiliary token indices
        self._start_idx = config["START_idx"]
        self._end_idx = config["END_idx"]
        self._pad_idx = config["PAD_idx"]
        self._UNK_idx = config["UNK_idx"]

        # Auxiliary token marks
        self._START_token = config["START_token"]
        self._END_token = config["END_token"]
        self._PAD_token = config["PAD_token"]
        self._UNK_token = config["UNK_token"]

        self._max_len = config["max_len"]

        # Transformation to apply to each image
        self._image_specs = config["image_specs"]
        self._image_transform = self._construct_image_transform(self._image_specs["image_size"])
    
    @property
    def SOS_ID(self):
        return self._start_idx
    
    @property
    def EOS_ID(self):
        return self._end_idx
    
    @property
    def PAD_ID(self):
        return self._pad_idx

    @property
    def IDX2WORD(self):
        return self._idx2word

    @property
    def max_len(self):
        return self._max_len
    
    @property
    def transform(self):
        return self._image_transform

    def _construct_image_transform(self, image_size=None):
        """Constructs the image preprocessing transform object.

        Arguments:
            image_size (int): Size of the result image
        """
        if not image_size:
            image_size = self._image_specs["image_size"]
        assert image_size, "Image size not provided."
        print(f"{image_size=}")

        # ImageNet normalization statistics
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

        return preprocessing

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            return cls(json.load(f))

class ImageCaptioningModel(torch.nn.Module):
    def __init__(self, config):
        super(ImageCaptioningModel, self).__init__()
        self.config = config
        self.encoder: ResNet
        self.decoder = CaptionDecoder(config)
        self.init_encoder()

    def init_encoder(self):
        #######################
        # Set up the encoder
        #######################
        # Download pretrained CNN encoder
        self.encoder = models.resnet50(pretrained=True)
        # Extract only the convolutional backbone of the model
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-2]))
        # Freeze encoder layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
    
    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(cls, path: str):
        """Load the model from a checkpoint dir."""

        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(config)
        with open(os.path.join(path, "model.pth"), "rb") as f:
            model.load_state_dict(torch.load(f))
        return model

    def save_pretrained(self, path: str):
        """Save the model to a checkpoint dir."""

        MODEL_SAVE_PREFIX = "model.pth"
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, MODEL_SAVE_PREFIX), "wb") as f:
            torch.save(self.state_dict(), f)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f)

    @classmethod
    def from_training_checkpoint(cls, checkpoint_path: str, config_path: str):
        """Load the model from a training checkpoint and config file.
        Note: Loads only the decoder part of the model. The encoder is loaded from pretrained weights.
        """

        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(config)
        with open(checkpoint_path, "rb") as f:
            model.decoder.load_state_dict(torch.load(f))
        return model

    def image_features(self, x: torch.Tensor):
        """Extract image features using the encoder."""
        x = x.to(self.device)

        # Extract image features
        img_features = self.encoder(x)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        img_features = img_features.detach()
        return img_features

    def load_image(self, image_path, image_transform):
        img_pil = Image.open(image_path).convert("RGB")
        image_tensor = image_transform(img_pil)
        return torch.stack([image_tensor], dim=0)

    def generate(self, image_path: str, tokenizer: ImageCaptioningTokenizer):
        self.eval()
        with torch.no_grad():
            image = self.load_image(image_path, tokenizer.transform)
            image_features = self.image_features(image)
            predictions = greedy_decoding(self.decoder, image_features, tokenizer.SOS_ID, tokenizer.EOS_ID, tokenizer.PAD_ID, tokenizer.IDX2WORD, tokenizer.max_len, self.device)
        return predictions




if __name__ == "__main__":
    # Test the model
    config_path = "/Users/saurabh/Documents/projects/image-captioning/config.json"
    model_checkpoint_path = "/Users/saurabh/Documents/projects/image-captioning/checkpoints/Dec-22_21-58-02/model_0.pth"

    load = torch.load(model_checkpoint_path)
    # for k, v in load.items():
    #     print(f"{k=}, {v.shape=}")
    model = ImageCaptioningModel.from_training_checkpoint(
        model_checkpoint_path, config_path
    )
    # print(f"{model=}")

    image_path = "/Users/saurabh/Documents/projects/image-captioning/dataset/flickr8k/Flickr8k_Dataset.zip/Flicker8k_Dataset/23445819_3a458716c1.jpg"

    tokenizer = ImageCaptioningTokenizer.from_config(config_path)
    print(f"{model.generate(image_path, tokenizer=tokenizer)=}")