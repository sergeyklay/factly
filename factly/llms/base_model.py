"""Common mixin for Factly LLM model implementations."""


class FactlyModelMixin:
    """Mixin providing common functionality for Factly LLM models.

    This mixin standardizes common operations across different model implementations
    such as message formatting and model name handling.
    """

    system_prompt: str
    actual_model_name: str

    @staticmethod
    def get_actual_model_name(model_name: str) -> str:
        """Extract base model name from provider-prefixed format.

        Args:
            model_name: Original model identifier, potentially in
                "<provider>/<model>" format

        Returns:
            The model name without provider prefix
        """
        if "/" in model_name:
            _, model_name = model_name.split("/", 1)
            return model_name
        return model_name

    def create_messages(self, prompt: str):
        """Format prompt into chat completion messages.

        Args:
            prompt: User input prompt

        Returns:
            List of message objects with appropriate roles
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        return messages

    def get_display_model_name(self) -> str:
        """Get model name for display in UI and reports.

        Returns:
            Model name suitable for display
        """
        return self.actual_model_name
