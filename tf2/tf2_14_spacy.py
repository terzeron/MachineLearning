#!/usr/bin/env python

import sys
import subprocess
import spacy

MODEL = "en_core_web_sm"

def ensure_pip():
    """가상환경에 pip가 없으면 설치(bootstrap)."""
    try:
        import pip  # noqa: F401
        return True
    except Exception:
        try:
            # python -m ensurepip --upgrade
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
            return True
        except Exception as e:
            print(f"[warn] ensurepip 실패: {e}")
            return False

        
def try_download_model():
    """모델 다운로드 시도: spacy CLI → (pip 없으면) ensurepip 후 재시도."""
    try:
        # 가장 간단한 경로: python -m spacy download en_core_web_sm
        subprocess.run([sys.executable, "-m", "spacy", "download", MODEL], check=True)
        return True
    except Exception as first_err:
        print(f"[warn] spaCy CLI 다운로드 실패(첫 시도): {first_err}")
        if ensure_pip():
            try:
                subprocess.run([sys.executable, "-m", "spacy", "download", MODEL], check=True)
                return True
            except Exception as second_err:
                print(f"[warn] spaCy CLI 다운로드 실패(재시도): {second_err}")
    return False


def load_nlp():
    try:
        return spacy.load(MODEL)
    except OSError:
        print(f"[info] '{MODEL}' 미설치: 다운로드 시도합니다...")
        if try_download_model():
            try:
                return spacy.load(MODEL)
            except Exception as e:
                print(f"[warn] 설치 후 로드 실패: {e}")

        # 최종 폴백: 가벼운 파이프라인(벡터/태깅 없음)
        print("[fallback] en_core_web_sm 없이 spacy.blank('en')을 사용합니다.")
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

nlp = load_nlp()

sentence = (
    "Natural language processing (NLP) is a subfield of computer science, "
    "information engineering, and artificial intelligence concerned with "
    "the interactions between computers and human (natural) languages, "
    "in particular how to program computers to process and analyze large "
    "amounts of natural language data."
)

doc = nlp(sentence)

word_tokenized_sentence = [token.text for token in doc]
print("word by word:", word_tokenized_sentence)

sentence_tokenized_list = [sent.text for sent in doc.sents]
print("line by line:", sentence_tokenized_list)
