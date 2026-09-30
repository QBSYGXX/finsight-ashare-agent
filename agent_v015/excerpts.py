"""模块：证据选句；作用：按原文位置编号、选择连续句段；核心业务：是。"""
import re


# 只识别明确的中文句末和独立小节标题；换行可能只是PDF排版，不当作句末。
END = re.compile(r'[。！？!?][”’」』）)]*')
HEADING = re.compile(r'(?m)^[ \t]*[（(][0-9一二三四五六七八九十]+[）)][^\n。！？]{1,60}\n')


def annotate(source, page):
    """句段必须完整落在已经观察的片段内，不自动扩展到未展示的页文本。"""
    text, offset = source['text'], source.get('start')
    if type(offset) is not int or offset < 0 or page['text'][offset:offset+len(text)] != text:
        return {**source, 'sentences': [], 'excerpt_boundary': 'unverified_offset'}
    page_text = page['text']
    spans = []
    cursor = 0
    # 标题重置起点，避免把上一节的句尾和下一节的正文拼在一起。
    boundaries = sorted([(m.start(), m.end(), 'end') for m in END.finditer(page_text)] +
                        [(m.start(), m.end(), 'heading') for m in HEADING.finditer(page_text)])
    for _, end, kind in boundaries:
        if kind == 'end':
            start = cursor
            while start < end and page_text[start].isspace(): start += 1
            if offset <= start < end <= offset+len(text) and end-start >= 12:
                spans.append({'id':len(spans), 'start':start-offset, 'end':end-offset,
                              'text':page_text[start:end]})
        cursor = end
    return {**source, 'sentences':spans,
            'excerpt_boundary':'has_sentence_spans' if spans else 'no_complete_sentence_in_excerpt'}


def select(source, first_sentence, last_sentence):
    """只接受本题工具返回的编号；保留连续原文及相对片段/整页两套坐标。"""
    spans = source.get('sentences', [])
    if type(first_sentence) is not int or type(last_sentence) is not int or not (
            0 <= first_sentence <= last_sentence < len(spans)):
        raise ValueError('Select valid consecutive sentence IDs from this observed evidence')
    start, end = spans[first_sentence]['start'], spans[last_sentence]['end']
    quote = source['text'][start:end]
    if not 12 <= len(quote) <= 1200:
        raise ValueError('Selected sentence span must contain 12..1200 characters; choose a smaller range')
    return {'quote':quote, 'match_mode':'selected_sentences', 'source_span':[start,end],
            'page_span':[source['start']+start,source['start']+end],
            'sentence_ids':list(range(first_sentence,last_sentence+1)),
            'boundary_check':'punctuation_and_section_heuristic',
            'source':{k:source[k] for k in ['id','company','report_year','report_id','page','pdf_sha256','source_url']}}
