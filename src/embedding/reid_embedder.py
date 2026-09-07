import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
import torchreid


OSNET_INPUT_HEIGHT = 256
OSNET_INPUT_WIDTH = 128

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]

DEFAULT_CHECKPOINT_PATH = os.path.join(
    "models",
    "osnet_x0_25_market1501.pth"
)


class ReIDEmbedder:
    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        device: str = "cpu",
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    ):
        """
        Build OSNet without automatic ImageNet weights and load
        the verified Market-1501-trained checkpoint.

        Args:
            model_name:
                TorchReID model name.

            device:
                Device used for inference. This project uses CPU.

            checkpoint_path:
                Path to the Market-1501-trained checkpoint.

        Raises:
            FileNotFoundError:
                If the checkpoint does not exist.

            RuntimeError:
                If the model or checkpoint cannot be loaded.
        """

        self.device = device

        # ---------------------------------------------------------
        # Check checkpoint existence
        # ---------------------------------------------------------

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Market-1501 checkpoint not found at "
                f"'{checkpoint_path}'. "
                f"Refusing to silently fall back to "
                f"ImageNet-pretrained weights."
            )

        # ---------------------------------------------------------
        # Build OSNet without automatic pretrained weights
        # ---------------------------------------------------------

        try:
            self._model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                pretrained=False,
            )

            # -----------------------------------------------------
            # Load verified Market-1501 checkpoint
            # -----------------------------------------------------

            torchreid.utils.load_pretrained_weights(
                self._model,
                checkpoint_path
            )

            self._model = self._model.to(
                self.device
            )

            self._model.eval()

        except Exception as e:
            raise RuntimeError(
                f"Failed to build '{model_name}' and load "
                f"checkpoint '{checkpoint_path}': {e}"
            ) from e

        # ---------------------------------------------------------
        # Preprocessing
        # ---------------------------------------------------------

        self._preprocess = T.Compose([
            T.ToPILImage(),

            T.Resize(
                (
                    OSNET_INPUT_HEIGHT,
                    OSNET_INPUT_WIDTH
                )
            ),

            T.ToTensor(),

            T.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])

    def embed(
        self,
        crop_bgr: np.ndarray
    ) -> np.ndarray:
        """
        Produce one L2-normalized Re-ID embedding.

        Args:
            crop_bgr:
                OpenCV BGR image with shape (H, W, 3).

        Returns:
            1D numpy.ndarray:
                shape: (512,)
                dtype: float32
                L2-normalized

        Raises:
            ValueError:
                If the crop is invalid.

            RuntimeError:
                If the model produces invalid output or
                a zero-norm embedding.
        """

        # ---------------------------------------------------------
        # Validate crop
        # ---------------------------------------------------------

        self._validate_crop(
            crop_bgr
        )

        # ---------------------------------------------------------
        # Convert BGR -> RGB
        # ---------------------------------------------------------

        crop_rgb = crop_bgr[:, :, ::-1]

        # ---------------------------------------------------------
        # Preprocess
        # ---------------------------------------------------------

        input_tensor = self._preprocess(
            crop_rgb
        ).unsqueeze(0).to(
            self.device
        )

        # ---------------------------------------------------------
        # Model inference
        # ---------------------------------------------------------

        with torch.no_grad():
            output = self._model(
                input_tensor
            )

        # ---------------------------------------------------------
        # Handle possible multiple outputs
        # ---------------------------------------------------------

        if isinstance(
            output,
            (tuple, list)
        ):
            output = output[0]

        # ---------------------------------------------------------
        # Convert embedding to numpy float32
        # ---------------------------------------------------------

        embedding = (
            output
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )

        # ---------------------------------------------------------
        # Validate finite values
        # ---------------------------------------------------------

        if not np.isfinite(
            embedding
        ).all():

            raise RuntimeError(
                "Embedding contains non-finite values "
                "(NaN/Inf) after inference."
            )

        # ---------------------------------------------------------
        # L2 normalization
        # ---------------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0.0:

            raise RuntimeError(
                "Embedding has zero norm; "
                "cannot L2-normalize "
                "(degenerate model output)."
            )

        embedding = (
            embedding / norm
        ).astype(np.float32)

        return embedding

    @staticmethod
    def _validate_crop(
        crop: Optional[np.ndarray]
    ) -> None:
        """
        Validate an OpenCV BGR crop.
        """

        if crop is None:
            raise ValueError(
                "Input crop is None."
            )

        if not isinstance(
            crop,
            np.ndarray
        ):
            raise ValueError(
                "Input crop must be a numpy array, "
                f"got {type(crop)}."
            )

        if crop.size == 0:
            raise ValueError(
                "Input crop is empty."
            )

        if (
            crop.ndim != 3
            or crop.shape[2] != 3
        ):
            raise ValueError(
                "Input crop must have shape "
                f"(H, W, 3), got shape {crop.shape}."
            )