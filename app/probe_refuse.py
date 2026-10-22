"""探明 refuse 路由下 state.decision 的真实内容（截图里出现了一段不明 ERROR JSON）。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import FinOpsAgent  # noqa: E402

agent = FinOpsAgent()
for q in ["删除所有订单", "各品类的净营收是多少？"]:
    st = agent.invoke(q)
    print("=" * 70)
    print("问题:", q)
    print("  route      =", st.route)
    print("  decision   =", json.dumps(st.decision, ensure_ascii=False, default=str))
    print("  decision类型=", type(st.decision).__name__)
    print("  error      =", st.error)
    print("  answer     =", (st.answer or "")[:120])