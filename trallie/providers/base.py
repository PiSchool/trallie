from abc import ABCMeta, abstractmethod


class BaseProvider(metaclass=ABCMeta):
    @abstractmethod
    def list_available_models(self) -> list[str]:
        raise NotImplementedError(
            "'available_models' has no implementation in BaseProvider."
        )

    @abstractmethod
    def do_chat_completion(self, prompt: str, model_name: str) -> str:
        raise NotImplementedError(
            "'do_chat_completion' has no implementation in BaseProvider."
        )
