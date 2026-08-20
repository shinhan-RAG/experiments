import unittest

from schema_adapter import adapt_tag
from structured_search import StructuredTagIndex, variants_of


class StructuredSearchTest(unittest.TestCase):
    def setUp(self):
        self.elements = [
            {"contract_scope": "암특약(무배당, 갱신형)"},
            {"contract_scope": "암특약(무배당, 해약환급금 미지급형)"},
            {"contract_scope": "뇌특약(무배당, 갱신형)"},
        ]
        self.tags = [
            {"contract_key": self.elements[0]["contract_scope"], "subject_key": ["표적항암"], "role": ["payment_trigger"],
             "locator": {"article": "제3조", "article_title": "보험금 지급사유"}, "schema_tag": "text"},
            {"contract_key": self.elements[1]["contract_scope"], "subject_key": ["표적항암"], "role": ["payment_trigger"],
             "locator": {"article": "제3조", "article_title": "보험금 지급사유"}, "schema_tag": "text"},
            {"contract_key": self.elements[2]["contract_scope"], "subject_key": ["뇌경색"], "role": ["code_reference"],
             "locator": {"article": "제8조", "article_title": "분류표", "table_headers": ["I63.9"]},
             "reference": ["제3조"], "schema_tag": "table"},
        ]
        self.idx = StructuredTagIndex(self.elements, self.tags)

    def test_variant_is_preserved(self):
        self.assertIn("갱신형", variants_of(self.elements[0]["contract_scope"]))
        ranked = self.idx.rank("암특약 갱신형 표적항암", {"contract": ["암특약"]})
        self.assertEqual(0, ranked[0][0])

    def test_table_and_article_are_searchable(self):
        ranked = self.idx.rank("I63.9 제8조 분류표", {"schema": ["table"]})
        self.assertEqual(2, ranked[0][0])

    def test_base_identity_keeps_variants_recallable(self):
        ranked = self.idx.rank("암특약 표적항암", {"contract": ["암특약"]})
        self.assertEqual({0, 1}, {ranked[0][0], ranked[1][0]})

    def test_unknown_schema_field_is_kept_as_low_weight_extra(self):
        tags = [
            {"element_id": "a", "novel_domain_key": "알파고유식별어"},
            {"element_id": "b", "novel_domain_key": "베타고유식별어"},
            {"element_id": "c", "novel_domain_key": "감마고유식별어"},
        ]
        idx = StructuredTagIndex(self.elements, tags)
        self.assertEqual(1, idx.rank("베타고유식별어", {}, limit=1)[0][0])
        self.assertEqual(3, idx.adapter_stats["documents_with_unknown"])

    def test_portable_business_document_keys_map_to_common_axes(self):
        row = adapt_tag({
            "document_type": "사업방법서", "product_name": "연금보험",
            "breadcrumb": ["10. 보험료에 관한 사항"], "conditions": ["월납"],
        })
        self.assertEqual(["연금보험"], row["identity"])
        self.assertEqual(["10. 보험료에 관한 사항"], row["locator"])
        self.assertEqual(["월납"], row["constraint"])
        self.assertEqual(["사업방법서"], row["structure"])

    def test_nested_locator_allows_more_specific_table_axis(self):
        row = adapt_tag({"locator": {"article": "제8조", "table_headers": ["질병코드"]}})
        self.assertEqual(["제8조"], row["locator"])
        self.assertEqual(["질병코드"], row["table"])


if __name__ == "__main__":
    unittest.main()
