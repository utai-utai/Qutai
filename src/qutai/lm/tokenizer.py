class Tokenizer:
    def __init__(self, text: str) -> None:
        self.characters: str = "".join(dict.fromkeys(text))
        self.indices: dict[str, int] = {character: index for index, character in enumerate(self.characters)}

    @property
    def size(self) -> int:
        """词表中的字符数量。"""
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        """把文本转换成整数编号，未知字符需要先加入建词表的文本。"""
        try:
            return [self.indices[character] for character in text]
        except KeyError as error:
            raise ValueError(f"词表中没有字符：{error.args[0]!r}") from None

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        """把整数编号还原为文本，不接受负数或词表范围外的编号。"""
        if any(type(index) is not int or not 0 <= index < self.size for index in ids):
            raise ValueError("字符编号必须是词表范围内的非负整数")
        return "".join(self.characters[index] for index in ids)
