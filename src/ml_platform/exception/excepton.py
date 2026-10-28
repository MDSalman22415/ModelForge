
import sys

from ml_platform.logging import logger


class ModelForge(Exception):

    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))

        self.error_message = error_message

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = None
            self.file_name = None

    def __str__(self):
        return (
            f"Error occurred in Python script "
            f"[{self.file_name}], line [{self.lineno}]: "
            f"{self.error_message}"
        )


if __name__ == "__main__":
    try:
        logger.info("Entering the try block")
        a = 1 / 0
        print(a)

    except Exception as e:
        raise ModelForge(e, sys) from e
