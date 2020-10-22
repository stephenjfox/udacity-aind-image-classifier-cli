from unittest import TestCase, main as test_main

from nn_trainer.utils import build_training_arg_parser


class TestAlgorithm(TestCase):
    def test_parsing_hyperparameterGroup(self):
        parser = build_training_arg_parser()
        test_learning_rate = 1e-3
        test_epochs = 10
        test_hidden_units = 128
        result = parser.parse_args([
            'testDirectory', '--learning_rate',
            str(test_learning_rate), '--epochs',
            str(test_epochs), '--hidden_units',
            str(test_hidden_units)
        ])
        self.assertEqual(result.learning_rate, test_learning_rate,
                         "hyperparameterGroup: learning rate")
        self.assertEqual(result.epochs, test_epochs, "hyperparameterGroup: epochs")
        self.assertEqual(result.hidden_units, test_hidden_units,
                         "hyperparameterGroup: hidden units")

    def test_parsing_dataDirectory(self):
        parser = build_training_arg_parser()
        input_directory = 'testDirectory'
        result = parser.parse_args([input_directory]).data_directory
        self.assertEqual(result, input_directory, "dataDirectory")

    def test_parsing_saveDirectory(self):
        parser = build_training_arg_parser()
        test_save_directory = 'exampleDirectory'
        result = parser.parse_args(['placeholderDirectory', '--save_dir',
                                    test_save_directory]).save_dir

        self.assertEqual(result, test_save_directory, "saveDirectory")

    def test_parsing_gpuDefault(self):
        subject = build_training_arg_parser().parse_args(['testDirectory'])
        test = subject.gpu
        self.assertFalse(test, 'parsing_gpuDefault should be False')


if __name__ == "__main__":
    test_main()
