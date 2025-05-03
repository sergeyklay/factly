class FactlyModelMixin:
    system_prompt: str
    actual_model_name: str

    @staticmethod
    def get_actual_model_name(model_name: str) -> str:
        """Get the actual model name without provider prefix."""
        if "/" in model_name:
            _, model_name = model_name.split("/", 1)
            return model_name
        return model_name

    def create_messages(self, prompt: str):
        """Create messages for the chat completion."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def get_display_model_name(self) -> str:
        """Get the display model name."""
        return self.actual_model_name
