from config import Configuration
from controller import Controller
from utils.parser import Parser


def main(args):
    config = Configuration(args)
    config.generate_config_dict()
    config.show_configuration()

    if config.running_mode == 'generate':
        dataset_generator = config.dataset_generator_class(config)
        dataset_generator.generate()
    elif config.running_mode == 'train':
        controller = Controller(config)
        controller.initialize()
        controller.config_dataset()
        controller.config_structure()
        controller.start()


if __name__ == '__main__':
    main(Parser().parse())
