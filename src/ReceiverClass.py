from threading import Thread
from flask import Flask, request, jsonify
from LoggerClass import Logger


class Receiver:
    """
    Receiver class for listening requests from mediator
    """
    def __init__(self, bot_url, port=5000):
        self.app = Flask(__name__)
        self.bot_url = bot_url
        self.port = port
        self.logger = Logger("Receiver")
        self.thread = None

        @self.app.route('/bot/turn', methods=["POST"])
        def get_current_game_field():
            current_game_field = request.json.get("game_field")
            self.logger.send_message(f"Запрос {request} был принят! Ответ сервера: {request.json}", "info")
            return jsonify({"game_field": current_game_field})

    def listen(self):
        """
        Start listening requests from mediator.
        :return:
        """
        try:
            self.thread = Thread(target=self.app.run, args=(self.bot_url, self.port, None))
            self.thread.start()
            self.logger.send_message("Поток для слушателя запущен успешно!", "info")
        except Exception as e:
            self.logger.send_message(f"Ошибка запуска потока: {e}", "error")
