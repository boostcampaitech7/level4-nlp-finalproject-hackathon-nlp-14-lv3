from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


class EmbeddingModel:
    _instance = None  # singleton 패턴 사용


    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


    def __init__(self):
        if not hasattr(self, "model"):
            self.model = None
    

    def load_model(self):
        if self.model is None:
            print("임베딩 모델 로드 중...")
            self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            print("모델 로드 완료")

    
    def get_embedding(self, sentence: str, max_length: int = 8192) -> list[float]:
        if self.model is None:
            raise RuntimeError("모델이 아직 로드되지 않았습니다. 먼저 load_model()을 실행하세요.")

        if isinstance(sentence, str):
            return self.model.encode(sentence, max_length=max_length)['dense_vecs']
        else:
            raise ValueError("입력은 문자열이어야 합니다.")


# if __name__ == "__main__":

#     print("임베딩 테스트")
#     print("종료는 ctrl+c")
#     embedding_model = EmbeddingModel()
#     embedding_model.load_model()

#     while(1):
#         sentence = input()
#         length = len(sentence) * 10
#         print(type(length), length)
#         print(f"임베딩 길이: {len(embedding_model.get_embedding(sentence, max_length=length))}\n")
