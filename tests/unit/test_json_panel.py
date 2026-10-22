"""`app.streamlit_app._as_json_object` 的回归测试。

背景：Streamlit 前端的 JSON 视图组件（`reactJsonViewCompat`）**只接受 JSON 对象**。
传 `None`（refuse / sql 路由下 `state.decision` 就是 `None`）或传数组
（`state.trace` 是 list）时，它会渲染成一个刺眼的

    {"ERROR": {"message": "src property must be a valid json object"}}

框，看上去像程序崩了。这个 helper 把所有输入收敛成 dict，避免误报。
"""

from __future__ import annotations

import pytest

from app.streamlit_app import _as_json_object


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"action": "metric"}, {"action": "metric"}),
        (None, {}),
        (["a", "b"], {"items": ["a", "b"]}),
        (("a",), {"items": ["a"]}),
        ("text", {"value": "text"}),
        (7, {"value": 7}),
        (False, {"value": False}),
    ],
)
def test_as_json_object_always_returns_dict(value, expected):
    got = _as_json_object(value)
    assert got == expected
    assert isinstance(got, dict)


def test_dict_passthrough_is_identity():
    """dict 必须原样透传，不能包一层，否则界面上的字段名会变。"""
    payload = {"arguments": {"metric": "revenue"}}
    assert _as_json_object(payload) is payload


def test_nested_non_serializable_value_kept_as_is():
    """只负责「外层是不是对象」，不负责深度转换——保持行为可预期。"""
    path = object()
    got = _as_json_object({"path": path})
    assert got["path"] is path