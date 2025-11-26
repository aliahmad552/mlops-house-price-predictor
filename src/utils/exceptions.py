import sys
from src.utils.logger import logger


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in script [{file_name}] "
        f"line [{line_number}] : {str(error)}"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        detailed_message = error_message_detail(error_message, error_detail)
        logger.error(detailed_message)
        super().__init__(detailed_message)
        self.error_message = detailed_message

    def __str__(self):
        return self.error_message
