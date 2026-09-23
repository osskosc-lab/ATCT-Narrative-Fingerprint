import unittest

from atct_fingerprint.document import parse_markdown_document
from atct_fingerprint.features import split_sentences


class MarkdownDocumentTests(unittest.TestCase):
    def test_prose_equations_and_structure_are_separated(self):
        text = """# 定義
これは第一の本文です。これは第二の本文です。

> これは引用であり本文層ではありません。
- これは箇条書きです。

## 仮説
$$
Z_{order} = 2
$$

これは第三の本文です。これは第四の本文です。

## 結論
これは第五の本文です。
"""
        parsed = parse_markdown_document(text, split_sentences)
        self.assertEqual(len(parsed.prose_sentences), 5)
        self.assertNotIn("Z_{order}", " ".join(parsed.prose_sentences))
        self.assertEqual(parsed.layer_counts["equation"], 1)
        self.assertEqual(parsed.layer_counts["heading"], 3)
        self.assertEqual(parsed.layer_counts["quote"], 1)
        equation = next(block for block in parsed.blocks if block.kind == "equation")
        self.assertEqual(equation.role, "hypothesis")
        self.assertEqual(
            [section.heading for section in parsed.sections],
            ["定義", "仮説", "結論"],
        )

    def test_numbered_heading_and_one_sentence_paragraph_are_preserved(self):
        text = """1. 魂の奥底を揺さぶる合図

そのノートは、この一年、開かれていない。
"""
        parsed = parse_markdown_document(text, split_sentences)
        heading = next(
            block for block in parsed.blocks if block.kind == "numbered_heading"
        )
        self.assertEqual(heading.text, "魂の奥底を揺さぶる合図")
        self.assertEqual(
            parsed.prose_sentences,
            ("そのノートは、この一年、開かれていない。",),
        )


if __name__ == "__main__":
    unittest.main()
