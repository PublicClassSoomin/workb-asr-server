import re
from kiwipiepy import Kiwi

kiwi = Kiwi()


def fix_spacing_with_kiwi(text: str) -> str:
    # 1. 여러 공백 정리
    text = re.sub(r"\s+", " ", text.strip())

    # 2. 한글 사이 공백 제거
    # 예: "우리 는" -> "우리는", "재밌 게" -> "재밌게"
    text = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", text)

    # 3. Kiwi로 띄어쓰기 복원
    text = kiwi.space(text)

    # 4. 문장부호 주변 공백 정리
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r"([,.?!])(?=[^\s])", r"\1 ", text)

    return text.strip()