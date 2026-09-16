from __future__ import annotations

import re

from context_router.domain import Relation, RouteRequest

#: Explicit cross-context cues. Kept as a family of natural phrasings rather than one
#: literal string, because "把 X 的做法用到 Y" and "把 X 应用到 Y" mean the same thing and
#: a rule that only knows the second turns every first into `unknown`.
_CROSS = re.compile(
    r"应用到|用到|用在|套用|复用|借鉴|迁移到|搬到|结合|关联|同时|两者|三者"
    r"|cross[- ]?context|apply\b.*?\bto\b|reuse|combine",
    re.I,
)
#: Explicit switch cues. Without these, "现在切换到 X" fell through to `unknown`, which
#: is the wrong label for a request the user stated outright.
_SWITCH = re.compile(r"切换到|转到|换成|换个(?:话题|方向|问题)|switch to|move on to", re.I)
_RETURN = re.compile(r"回到|回头|之前那个|前面那个|return to|go back", re.I)
_NEW = re.compile(r"全新|新项目|开始研究|另一个项目|new project|start (?:a )?new", re.I)
#: Bare demonstratives continue the current context only when they point at something that
#: exists. "那个尚未讨论的部署密钥" points at nothing, so it is not a continuation.
_ABSENT = r"(?!尚未|还没|没有|未曾|未讨论|未提到)"
_CONTINUE = re.compile(
    rf"继续|接着|刚才|这个{_ABSENT}|那个{_ABSENT}|它|continue|that one|\bit\b",
    re.I,
)


class RuleRelationClassifier:
    """High-confidence relation cues only; everything else stays `unknown`.

    The plan reserves fuzzy relations for a small model, so this classifier is
    deliberately conservative. `unknown` is a real answer: the selection policy widens on
    it rather than guessing a single context, which keeps a rule miss from silently
    discarding evidence.
    """

    model_version = "relation-rules-v2"

    def classify(self, request: RouteRequest) -> tuple[Relation, dict[str, float]]:
        query = request.query
        relation: Relation
        confidence: float
        if _CROSS.search(query):
            relation, confidence = "cross_context", 0.88
        elif _SWITCH.search(query):
            relation, confidence = "switch_or_return", 0.88
        elif _RETURN.search(query):
            relation, confidence = "switch_or_return", 0.88
        elif _NEW.search(query):
            relation, confidence = "new_context", 0.86
        elif _CONTINUE.search(query):
            relation, confidence = "continue", 0.84
        else:
            relation, confidence = "unknown", 0.58
        labels: list[Relation] = [
            "continue",
            "switch_or_return",
            "cross_context",
            "new_context",
            "unknown",
        ]
        remainder = (1.0 - confidence) / (len(labels) - 1)
        return relation, {label: confidence if label == relation else remainder for label in labels}
