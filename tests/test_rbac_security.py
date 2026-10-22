"""Test hồi quy cho các bản sửa bảo mật P0.

Phạm vi (không cần DB/Ollama):
- vectorstore._where_sql: allow-list category -> `category = ANY(%s)`; rỗng -> FALSE.
- retrieval._resolve_category_filter: RBAC least privilege.
- retrieval.retrieve/bm25_search: từ chối category ngoài quyền, fail-closed khi lỗi.
- cache.make_key: allow-list nằm trong key (không chia sẻ đáp án chéo quyền).
- api._user_allowed_categories: suy ra allow-list từ user.

Chạy: python -m unittest discover -s tests -v
"""
import importlib
import unittest


class TestWhereSQL(unittest.TestCase):
    def setUp(self):
        self.vs = importlib.import_module("backend.indexing.vectorstore")

    def test_category_list_uses_any(self):
        sql, params = self.vs._where_sql({"category": ["a", "b"]})
        self.assertIn("category = ANY(%s)", sql)
        self.assertEqual(params, [["a", "b"]])

    def test_category_empty_list_denies(self):
        sql, params = self.vs._where_sql({"category": []})
        self.assertEqual(sql, "FALSE")
        self.assertEqual(params, [])

    def test_scalar_category_unchanged(self):
        sql, params = self.vs._where_sql({"category": "quy_che"})
        self.assertIn("category = %s", sql)
        self.assertNotIn("ANY", sql)
        self.assertEqual(params, ["quy_che"])


class TestCategoryFilter(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module("backend.indexing.retrieval")

    def test_unrestricted_when_no_allowlist(self):
        self.assertEqual(self.r._resolve_category_filter(None, None), (True, None))

    def test_star_is_unrestricted(self):
        self.assertEqual(self.r._resolve_category_filter("quy_che", ["*"]), (True, "quy_che"))

    def test_deny_category_outside_allowlist(self):
        ok, _ = self.r._resolve_category_filter("quy_che", ["thong_bao"])
        self.assertFalse(ok)

    def test_allowlist_used_when_category_missing(self):
        ok, filt = self.r._resolve_category_filter(None, ["thong_bao", "quy_che"])
        self.assertTrue(ok)
        self.assertEqual(sorted(filt), ["quy_che", "thong_bao"])


class TestRetrieveRBAC(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module("backend.indexing.retrieval")
        self._orig_embed = self.r.embed
        self._orig_qv = self.r.query_vector

    def tearDown(self):
        self.r.embed = self._orig_embed
        self.r.query_vector = self._orig_qv

    def test_denied_category_returns_empty_without_embedding(self):
        called = {"embed": False}

        def fake_embed(texts):
            called["embed"] = True
            return [[0.1, 0.2]]

        self.r.embed = fake_embed
        out = self.r.retrieve("q", category="quy_che", allowed_categories=["thong_bao"])
        self.assertEqual(out, [])
        self.assertFalse(called["embed"], "không được embed khi user không có quyền category")

    def test_allowlist_forwarded_to_vector_query(self):
        captured = {}
        self.r.embed = lambda texts: [[0.1, 0.2]]

        def fake_qv(q_emb, where=None, n=5):
            captured["where"] = where
            return []

        self.r.query_vector = fake_qv
        self.r.retrieve("q", category=None, allowed_categories=["thong_bao", "quy_che"])
        self.assertEqual(captured.get("where"), {"category": ["thong_bao", "quy_che"]})

    def test_fail_closed_on_vector_error(self):
        self.r.embed = lambda texts: [[0.1, 0.2]]

        def boom(*a, **k):
            raise RuntimeError("db down")

        self.r.query_vector = boom
        out = self.r.retrieve("q", category="quy_che")
        self.assertEqual(out, [], "lỗi truy vấn phải fail-closed, không quét toàn bộ index")

    def test_fail_closed_on_session_vector_error(self):
        self.r.embed = lambda texts: [[0.1, 0.2]]

        def boom(*a, **k):
            raise RuntimeError("db down")

        self.r.query_vector = boom
        out = self.r.retrieve("q", session_id="sess-1")
        self.assertEqual(out, [], "lỗi phiên phải fail-closed, không rò rỉ session khác")

    def test_bm25_denies_category(self):
        out = self.r.bm25_search("q", category="quy_che", allowed_categories=["thong_bao"])
        self.assertEqual(out, [])


class TestCacheKey(unittest.TestCase):
    def setUp(self):
        self.cache = importlib.import_module("backend.infra.cache")

    def test_different_allowlists_produce_different_keys(self):
        k1 = self.cache.make_key("q", None, 5, "formal", ["thong_bao"])
        k2 = self.cache.make_key("q", None, 5, "formal", ["quy_che"])
        self.assertNotEqual(k1, k2)

    def test_none_and_empty_scope_match(self):
        k1 = self.cache.make_key("q", None, 5, "formal", None)
        k2 = self.cache.make_key("q", None, 5, "formal", [])
        self.assertEqual(k1, k2)


class TestUserAllowedCategories(unittest.TestCase):
    def setUp(self):
        self.app_mod = importlib.import_module("backend.api.app")
        self.cfg = importlib.import_module("backend.config.settings")

    def _with_auth(self, enabled):
        orig = self.cfg.AUTH_ENABLED
        self.cfg.AUTH_ENABLED = enabled
        return orig

    def test_auth_disabled_returns_none(self):
        orig = self._with_auth(False)
        try:
            self.assertIsNone(self.app_mod._user_allowed_categories({"categories": ["a"]}))
        finally:
            self.cfg.AUTH_ENABLED = orig

    def test_star_returns_none(self):
        orig = self._with_auth(True)
        try:
            self.assertIsNone(self.app_mod._user_allowed_categories({"categories": ["*"]}))
        finally:
            self.cfg.AUTH_ENABLED = orig

    def test_restricted_returns_list(self):
        orig = self._with_auth(True)
        try:
            self.assertEqual(
                self.app_mod._user_allowed_categories({"categories": ["thong_bao"]}),
                ["thong_bao"],
            )
        finally:
            self.cfg.AUTH_ENABLED = orig


class TestFusion(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module("backend.indexing.retrieval")

    def test_normalize_distance_lower_is_better(self):
        docs = [{"score": 0.1}, {"score": 0.9}]
        n = self.r._normalize_scores(docs, is_distance=True)
        self.assertGreater(n[0], n[1])

    def test_normalize_bm25_higher_is_better(self):
        docs = [{"score": 5.0, "bm25_score": 5.0}, {"score": 1.0, "bm25_score": 1.0}]
        n = self.r._normalize_scores(docs, is_distance=False)
        self.assertGreater(n[0], n[1])

    def test_rrf_prefers_doc_in_both_branches(self):
        vec = [{"text": "a", "metadata": {}, "score": 0.1},
               {"text": "b", "metadata": {}, "score": 0.2}]
        bm = [{"text": "b", "metadata": {}, "score": 5.0},
              {"text": "c", "metadata": {}, "score": 3.0}]
        out = self.r._rrf_fuse(vec, bm, top_k=3, rrf_k=60)
        self.assertEqual(out[0]["text"], "b")

    def test_hybrid_uses_rrf_when_configured(self):
        orig_retrieve = self.r.retrieve
        orig_bm25 = self.r.bm25_search
        orig_fusion = self.r.RETR_CFG.get("fusion")
        try:
            self.r.RETR_CFG["fusion"] = "rrf"
            self.r.retrieve = lambda *a, **k: [
                {"text": "a", "metadata": {}, "score": 0.1},
                {"text": "b", "metadata": {}, "score": 0.2},
            ]
            self.r.bm25_search = lambda *a, **k: [
                {"text": "b", "metadata": {}, "score": 5.0},
                {"text": "c", "metadata": {}, "score": 3.0},
            ]
            out = self.r.hybrid_retrieve("q", top_k=3)
            self.assertEqual(out[0]["text"], "b")
        finally:
            self.r.retrieve = orig_retrieve
            self.r.bm25_search = orig_bm25
            if orig_fusion is None:
                self.r.RETR_CFG.pop("fusion", None)
            else:
                self.r.RETR_CFG["fusion"] = orig_fusion


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = importlib.import_module("backend.config.settings")

    def test_fusion_default_weighted(self):
        self.assertIn(self.cfg.RETRIEVAL.get("fusion"), ("weighted", "rrf"))

    def test_rerank_config_present(self):
        self.assertTrue(self.cfg.RERANK_MODEL)
        self.assertIsInstance(self.cfg.RERANK_HEURISTIC_BOOST, bool)


class TestConcurrency(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module("backend.infra.concurrency")

    def test_zero_limit_does_not_block(self):
        with self.c.limit("unit-zero", 0):
            pass

    def test_serializes_with_limit_one(self):
        import threading
        import time

        active = {"n": 0, "max": 0}
        lock = threading.Lock()

        def work():
            with self.c.limit("unit-serial", 1):
                with lock:
                    active["n"] += 1
                    active["max"] = max(active["max"], active["n"])
                time.sleep(0.02)
                with lock:
                    active["n"] -= 1

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(active["max"], 1)

    def test_timeout_raises(self):
        sem = self.c.get_semaphore("unit-timeout", 1)
        sem.acquire()
        try:
            with self.assertRaises(TimeoutError):
                with self.c.limit("unit-timeout", 1, timeout=0.01):
                    pass
        finally:
            sem.release()


if __name__ == "__main__":
    unittest.main()
