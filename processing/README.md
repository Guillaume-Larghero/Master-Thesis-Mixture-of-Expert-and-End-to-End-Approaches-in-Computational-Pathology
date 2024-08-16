# Processing Utils

Follow this guide to use our processing scripts.

**Basics:** Create a new conda environment and install the requirements. If you use `CUDA 12.X`, make sure to install the correct `onnxruntime-gpu` version. See [here](https://onnxruntime.ai/docs/install/#python-installs) for more information.

## Create Tiles

TODO: add docs

## Create Features

If you intend to use the `UNI` model, you will need to request access to the model parameters first ([here](https://huggingface.co/MahmoodLab/UNI)). Then add your `hf_token` from [here](https://huggingface.co/settings/tokens) to the `slurm_features.sh` script.

For `ctrans` and `chief` make sure you downloaded the onnx-exported model files. If you don't have them, get in touch with a member of the team. The `slurm_features.sh` script expects them to be in these locations:

```bash
CHIEF_PTH="/home/che099/models/chief.onnx"
CTRANS_PTH="/home/che099/models/ctranspath.onnx"
```

The other models are downloaded on the fly in case you want to use them. Currently we support the following models. We may add more support over time.

- chief (requires `onnxruntime`)
- ctrans (requires `onnxruntime`)
- phikon
- swin224
- resnet50
- lunit
- uni (requires `hf_token`)

For faster processing, it makes sense to split your dataset into parts. You can use a loop to do so quickly.

```bash
n_parts=50
for i in $(seq 0 $((n_parts - 1))); do
    sbatch slurm_features.sh /to/patches /to/features $n_parts $i
    sleep 0.1
done
```
