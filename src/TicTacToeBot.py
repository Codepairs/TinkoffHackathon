import requests
import os
from dotenv import load_dotenv, find_dotenv
from json import dumps, loads
from LoggerClass import Logger
from ReceiverClass import Receiver
from SolverClass import Solver
load_dotenv(find_dotenv())


class Bot:
    """
    Bot class for connect to mediator and play tic-tac-toe game
    """
    def __init__(self):

        self._logger = Logger('bot')
        self.session_id = self._get_session_id()
        self.bot_url = self._get_bot_url()
        self.mediator_url = self._get_mediator_url()
        self.receiver = Receiver(self.bot_url)
        self.thread = None
        self.bot_id = '123'
        self.password = '123'
        #self.figure = self.registration_request()
        self.starting_game_field = '_' * 361
        self.current_game_field = self.starting_game_field
        self.solver = Solver(self.current_game_field)

    @staticmethod
    def _get_session_id():
        """
        Get session_id from .env
        :return: session_id
        """
        session_id = os.getenv('SESSION_ID')
        if session_id is None:
            raise ValueError('SESSION_ID is not set')
        return session_id

    @staticmethod
    def _get_bot_url():
        """
        Get bot_url from .env
        :return: bot_url
        """
        bot_url = os.getenv('BOT_URL')
        if bot_url is None:
            raise ValueError('BOT_URL is not set')
        return bot_url

    @staticmethod
    def _get_mediator_url():
        """
        Get mediator_url from .env
        :return: mediator_url
        """
        mediator_id = os.getenv('MEDIATOR_ID')
        if mediator_id is None:
            raise ValueError('MEDIATOR_URL is not set')
        return mediator_id

    def registration_request(self):
        """
        Registation request. Response is a Figure to play.
        :return: figure
        """
        data_to_register = dumps({"bot_id": self.bot_id,
                            "password": self.password,
                            "bot_url": self.bot_url})
        self._logger.send_message(data_to_register, 'info')
        self._logger.send_message(f'{self.mediator_url}/sessions/{self.session_id}/registration', 'info')
        figure_response = requests.post(f'{self.mediator_url}/sessions/{self.session_id}/registration', data=data_to_register)
        if not figure_response.ok:
            raise requests.RequestException(f'Registration request failed: {figure_response}')
        self._logger.send_message(figure_response.text, 'info')
        raw_response = loads(figure_response.content)
        return raw_response['figure']

    def turn_request(self) -> str:
        """
        Turn request. Response is a new game field after your turn
        :return: new game field send to mediator
        """
        extension = '/bot/turn'
        # Calculate new turn
        suggested_turn = self.solver.make_turn()
        data_to_request = {"game_field": f'{suggested_turn}'}
        # Send turn and receive game_field after this turn
        new_field_response = requests.post(self.mediator_url + extension, json=data_to_request)
        if not new_field_response.ok:
            raise requests.RequestException(f'Turn request failed: {new_field_response}')
        new_field = loads(new_field_response.content)['game_field']
        self.current_game_field = new_field
        return new_field


bot = Bot()
bot.receiver.listen()
print("Запуск бота")
for i in range(40):
    bot.solver.logger.send_message(f'Ход {i+1}', 'info')
    bot.current_game_field = bot.turn_request()
    bot.solver.output(bot.current_game_field)
