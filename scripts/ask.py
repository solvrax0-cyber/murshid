"""Interactive CLI. Usage: python -m scripts.ask   (type 'جديد' for a new conversation)"""
import uuid

from app import store
from app.index import HybridIndex
from app.rag import RAGPipeline

if __name__ == "__main__":
    store.init()
    pipe = RAGPipeline(HybridIndex.load())
    sid = uuid.uuid4().hex
    print("مُرشِد جاهز. اكتب سؤالك (exit للخروج، جديد لمحادثة جديدة)")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        if q == "جديد":
            sid = uuid.uuid4().hex
            continue
        if not q:
            continue
        a = pipe.answer(q, sid)
        print(f"\n{a.answer}\n")
        for c in a.citations:
            print(f"  [{c['n']}] {c['label']}")
        print(f"  ({a.mode}, {a.latency_ms} ms, صياغة: {a.rewritten})")
