"""Shared semantic guard, independent of model and thinking policy."""
import re


# 模块：比率同比口径；作用：执行前要求明确增长率或百分点；核心业务：是。
def ambiguous_ratio_yoy(text, mentions, registry):
    """Return affected metric IDs. Only registered metric aliases are recognized.

    Keep each metric's suffix separate: a growth-rate request for revenue must
    not authorize a different ratio's unspecified YoY. Explicit group pronouns
    bind the final suffix to the whole group, matching the pinned planner.
    Unknown phrasings remain subject to the original planner's validation.
    """
    if not mentions:
        return []
    group = any(word in text for word in ("它们", "他们", "各自"))
    shared = text[mentions[-1][1]:]
    ambiguous = []
    for i, (start, end, metric) in enumerate(mentions):
        if registry[metric]["unit"] != "%":
            continue
        suffix = shared if group else text[end:mentions[i+1][0] if i+1 < len(mentions) else len(text)]
        # Prefix form: '同比毛利率' or '同比增长率：毛利率'. Do not inherit
        # a previous metric's suffix as this metric's prefix.
        prefix = text[:start] if i == 0 else ""
        prefix_match = re.search(r"同比(?:增长率|增长百分比|变化多少个百分点|百分点变化)?[\s：:的]*$", prefix)
        qualifier = suffix + (prefix_match.group(0) if prefix_match else "")
        if "同比" not in qualifier:
            continue
        rate = any(word in qualifier for word in ("增长率", "增长百分比"))
        pp = "百分点" in qualifier
        if rate == pp:  # Neither specified, or contradictory definitions.
            ambiguous.append(metric)
    return ambiguous
