"""
Predicting classes
* The predict.py script successfully reads in an image and a checkpoint then prints
  the most likely image class and it's associated probability.

Top K classes
* The predict.py script allows users to print out the top K classes along
  with associated probabilities.

Displaying class names
* The predict.py script allows users to load a JSON file that maps the class
  values to other category names.

Predicting with GPU
* The predict.py script allows users to use the GPU to calculate the predictions.

"""

from argparse import ArgumentParser, Namespace

from PIL import Image
from nn_trainer.model_loading import DEFAULT_TRANSFORMS, PersistableNet

import torch
from torch import nn


def parse_arguments() -> Namespace:
    parser = ArgumentParser("Image Classifier Training Program",
                            description="Train your own classifier that works on your images!")

    parser.add_argument('image_path',
                        help='Path to an image which can be loaded via PIL and '
                        'inferenced against.',
                        type=str)

    parser.add_argument('checkpoint', help='Model checkpoint to load for inference', type=str)

    parser.add_argument('--category_names',
                        help='Mapping between predicted indices and named classes'
                        'i.e. instead of "1" as an output, get "Russian lynx"',
                        type=str)

    parser.add_argument('--top_k',
                        help='The first k-most likely predictions. '
                        'Arguments less than or equal to 0 produce undefined behavior',
                        default=0,
                        type=int)

    parser.add_argument('--gpu', help='Enable GPU-based inference', action='store_true')

    args = parser.parse_args()
    if not isinstance(args.top_k, int):
        raise ValueError('Top-K argument should be valid integer')

    if args.top_k < 0:
        raise ValueError('Top-K argument should be positive integer')

    return args


def run_prediction(image_path, model, top_k=5, use_gpu: bool = False):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    im = Image.open(image_path)
    input_ = DEFAULT_TRANSFORMS[-1].transform(im)
    # should the model stay GPU-only mode or be put to CPU? This is a deployment question...

    with torch.no_grad():
        output = model(input_.unsqueeze(0))

    idx = model._image_index['idx_to_class']

    # my implementation
    if top_k:
        probs, preds = torch.topk(output, k=top_k, dim=-1)
        print("Predictions", preds.squeeze())
        pred_classes = [idx[i] for i in preds.squeeze()]
    else:
        prob, pred = torch.max(output, dim=-1)
        probs, pred_classes = prob.unsqueeze(0), [idx[pred]]

    # logits are actually treated with the negative log likelihood before loss calculation
    # need to apply a softmax
    return nn.functional.softmax(probs, dim=-1).squeeze(), pred_classes


def main():
    args = parse_arguments()
    # handle pushing to device
    model = PersistableNet.load(args.checkpoint).model
    # image = load_image_as_tensor(args.image_path)
    # use GPU
    probabilities, predictions = run_prediction(args.image_path,
                                                model,
                                                args.top_k,
                                                use_gpu=args.gpu)

    print("Probalities:", probabilities)
    print("Predictions:", predictions)

    if args.category_names:
        prediction_to_category = load_mapping(args.category_names)  # :: index int -> str
        output = list(map(prediction_to_category, predictions))
    else:
        output = predictions

    for o, p in zip(output, probabilities):
        formatted = f"{o:20}" if args.category_names else f"{o:>4}"
        print(f"Found class {formatted} with probability {p:0.4f}")


if __name__ == "__main__":
    main()