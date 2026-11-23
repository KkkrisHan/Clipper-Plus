import unittest

from videomind.transcript_segmentation import segment_transcript
from videomind.transcription import format_markdown


def words(items):
    return [{"start": start, "end": end, "text": text} for start, end, text in items]


class TranscriptSegmentationTests(unittest.TestCase):
    def test_fragments_across_whisper_segments_join_at_sentence_end(self):
        result = segment_transcript([
            {"start": 0, "end": 1, "text": "Today I want"},
            {"start": 1, "end": 2, "text": "to introduce"},
            {"start": 2, "end": 3, "text": "my new project."},
        ])
        self.assertEqual(result, [{"start": 0, "end": 3, "text": "Today I want to introduce my new project."}])

    def test_two_sentences_in_segment_use_word_start_times(self):
        result = segment_transcript([{"start": 0, "end": 5, "text": "Hello everyone. Today we will learn Python.", "words": words([
            (0.0, .4, " Hello"), (.4, .8, " everyone."), (1.2, 1.5, " Today"),
            (1.5, 1.7, " we"), (1.7, 2.0, " will"), (2.0, 2.4, " learn"), (2.4, 3.0, " Python."),
        ])}])
        self.assertEqual([(item["start"], item["text"]) for item in result], [
            (0.0, "Hello everyone."), (1.2, "Today we will learn Python.")])

    def test_mixed_chinese_english_spacing(self):
        result = segment_transcript([
            {"start": 0, "end": 1, "text": "今天我们介绍"},
            {"start": 1, "end": 2, "text": "Python"},
            {"start": 2, "end": 3, "text": "这个编程语言。"},
        ])
        self.assertEqual(result[0]["text"], "今天我们介绍 Python 这个编程语言。")

    def test_abbreviation_and_decimal_do_not_break_sentence(self):
        result = segment_transcript([{"start": 0, "end": 5, "text": "Dr. Smith explained version 3.14.", "words": words([
            (0, .2, " Dr."), (.2, .5, " Smith"), (.5, .9, " explained"),
            (.9, 1.0, " version"), (1.0, 1.2, " 3.14."),
        ])}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Dr. Smith explained version 3.14.")

    def test_long_unpunctuated_phrase_is_never_cut_arbitrarily(self):
        text = "a long continuous phrase without a reliable boundary"
        result = segment_transcript([{"start": 0, "end": 24, "text": text}])
        self.assertEqual([item["text"] for item in result], [text])

    def test_pause_threshold_and_sentence_boundary(self):
        fragments = [
            {"start": 0, "end": 1, "text": "this continues"},
            {"start": 1.5, "end": 2, "text": "without punctuation"},
        ]
        self.assertEqual(len(segment_transcript(fragments)), 1)  # weak pause does not fracture a clause
        fragments[0]["end"] = 21
        fragments[1]["start"] = 22.4
        self.assertEqual(len(segment_transcript(fragments)), 2)  # reliable long pause

    def test_text_order_and_content_are_preserved(self):
        source = [
            {"start": 0, "end": 1, "text": "alpha beta"},
            {"start": 1, "end": 2, "text": "gamma delta."},
        ]
        grouped = segment_transcript(source)
        self.assertEqual(" ".join(item["text"] for item in grouped), "alpha beta gamma delta.")

    def test_word_level_content_is_preserved_and_markdown_timestamp_is_original(self):
        result = segment_transcript([{"start": 0, "end": 4, "text": "Hi. There!", "words": words([
            (0.2, .5, " Hi."), (2.3, 2.8, " There!"),
        ])}])
        markdown = format_markdown({"title": "Example", "url": "https://example.com/v"}, result)
        self.assertIn("[00:00:00] Hi.", markdown)
        self.assertIn("[00:00:02] There!", markdown)


if __name__ == "__main__":
    unittest.main()
