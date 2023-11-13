from LoggerClass import Logger


def main():
    main_logger = Logger('Main')
    main_logger.send_message('Hello, World!', 'info')


if __name__ == '__main__':
    main()
