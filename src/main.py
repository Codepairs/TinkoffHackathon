from LoggerClass import Logger
from FigureClass import Figure


def main():
    main_logger = Logger('Main')
    main_logger.send_message('Hello, World!', 'info')
    figure = Figure()
    print(figure.get_cross_out_figure('f'))


if __name__ == '__main__':
    main()
