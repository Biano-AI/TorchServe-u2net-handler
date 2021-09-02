"""
Module for image classification default handler
"""
from __future__ import print_function, division
import logging
import io
import os
import torch
import time
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import base64
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class U2NetHandler(BaseHandler):

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
       First try to load torchscript else load eager mode state_dict based model.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
        else:
            logger.debug("Loading torchscript model")
            self.model = self._load_torchscript_model(model_pt_path)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)
        self.mapping = {}

        self.initialized = True

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a PyTorch model,
         returns an Numpy array
        """
        normalize = Compose([
            Resize((320, 320)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        return torch.stack([normalize(im) for im in data])

    def _get_mask_bytes(self, img, mask):
        logger.info(img.size)
        return Image.fromarray(mask).resize(img.size, Image.LANCZOS).tobytes()

    def postprocess(self, images, output):
        pred = output[0][:, 0, :, :]
        predict = self._normPRED(pred)
        predict_np = predict.cpu().detach().numpy()
        logger.info(f'predict_np shape {predict_np.shape}')
        res = []
        i = 0
        for im in images:
            logger.info(f'postprocessing image {i}')
            mask = (predict_np[i] * 255).astype(np.uint8)
            res.append(self._get_mask_bytes(im, mask))
        return res

    # normalize the predicted SOD probability map
    # from oficial U^2-Net repo
    def _normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def load_images(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = Image.open(io.BytesIO(image))
            images.append(image)
        return images

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        images = self.load_images(data)
        data_preprocess = self.preprocess(images)

        if not self._is_explain():
            output = self.inference(data_preprocess)
            output = self.postprocess(images, output)
        else:
            output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output
