import json


class NotCorrectFigure(Exception):
    def __init__(self, message):
        self.message = message


class Figure:
    def __init__(self, name):
        self.name = name
        self._cross = 'x'
        self._zero = 'o'
        self._crossed_out_cross = 'X'
        self._crossed_out_zero = 'O'
        self._empty = '_'

    def get_name(self):
        return self.name

    def get_cross_out_figure(self, figure):
        if figure == self._cross:
            return self._crossed_out_cross
        elif figure == self._zero:
            return self._crossed_out_zero
        else:
            raise NotCorrectFigure('Invalid _figure')

    def get_opposite_figure(self, figure):
        if figure == self._cross:
            return self._zero
        elif figure == self._zero:
            return self._cross
        else:
            raise NotCorrectFigure('Invalid _figure')



