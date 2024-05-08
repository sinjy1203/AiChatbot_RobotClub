import re


def get_response_dict(response_text):
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": response_text}}]},
    }


def del_prefix(text, prefix):
    pattern = "^" + re.escape(prefix)
    return re.sub(pattern, "", text).strip()
