from app.schemas import ChatMessage


def build_messages(messages: list[ChatMessage], system_prompt: str) -> list[ChatMessage]:
    user_visible_messages = [message for message in messages if message.role != "system"]
    return [ChatMessage(role="system", content=system_prompt), *user_visible_messages]
