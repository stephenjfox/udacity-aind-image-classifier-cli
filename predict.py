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

import torch


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


def main():
    args = parse_arguments()
    # handle pushing to device
    model = load_model(args.checkpoint)
    image = load_image_as_tensor(args.image_path)
    # use GPU
    prediction = run_prediction(model, image, use_gpu=args.gpu)

    if args.top_k:
        probs, preds = torch.top_k(prediction, k=args.top_k, dim=-1)
    else:
        prob, pred = torch.max(prediction, dim=-1)
        probs, preds = [prob], [pred]

    if args.category_names:
        prediction_to_category = load_mapping(args.category_names)  # :: index int -> str
        output = list(map(prediction_to_category, preds))
    else:
        output = preds

    for o in zip(output, probs):
        print(o)


if __name__ == "__main__":
    main()