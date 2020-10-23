# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## Instructions to run (for grader)

Follows the criteria of the implementation, as articulated by Udacity
(and outlined at the tops of `predict.py` and `train.py`), in a heavily
object-oriented way.
* Uses types abundantly
* Heavily based on the strong abstractions discovered in the notebook

You can find the exported html in `Image Classifier Project.html`

Some of the workspace instances that I encountered in the Udacity environment failed to correctly path to files, somewhat arbitrarily. So keep an eye out for that.



## Unit Test Coverage

I wrote unit tests to establish the floor on the core utilities in the
  project before just grinding out the implementation.

To run the tests (assuming Python 3)

* `python -m unittest nn_trainer.<submodule>`
  * Example: `python -m unittest nn_trainer.utils_test`