# U^2-Net TorchServe custom handler 

Download pretrained model [U^2-net](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing) or [U^2-netp](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing).

### 1. create mar file
```bash
docker run --rm -it --name mar -v $(pwd)/output:/output -v \
$(pwd)/model:/model -v $(pwd)/src/:/src pytorch/torchserve:latest \
torch-model-archiver --model-name u2net --version ${MODEL_VERSION:-'1.0'} \
--model-file /src/u2net.py \
--serialized-file /model/u2net.pth --export-path /output \
--extra-files /src/unet_classes.py --handler /src/custom_handler.py
```

### 2. Run TorchServe 
```bash
docker run --rm -it -v $(pwd)/output:/home/model-server/model-store \
-v $(pwd)/config.properties:/tmp/config.properties \
-p 8080:8080 -p 8081:8081 -p 8082:8082 pytorch/torchserve:latest \
torchserve --start --model-store model-store --ts-config /tmp/config.properties
```
