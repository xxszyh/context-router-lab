from __future__ import annotations

import re

from context_router.domain import Relation, RouteRequest

_CROSS = re.compile(r"应用到|结合|关联|同时|两者|三者|cross[- ]?context|apply.+to|combine", re.I)
_RETURN = re.compile(r"回到|回头|之前那个|前面那个|return to|go back", re.I)
_CONTINUE = re.compile(r"继续|刚才|这个|那个|它|接着|continue|that one|it\b", re.I)
_NEW = re.compile(r"全新|新项目|开始研究|另一个项目|new project|start (?:a )?new", re.I)


class RuleRelationClassifier:
    model_version = "relation-rules-v1"

    def classify(self, request: RouteRequest) -> tuple[Relation, dict[str, float]]:
        query = request.query
        relation: Relation
        confidence: float
        if _CROSS.search(query):
            relation, confidence = "cross_context", 0.88
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
