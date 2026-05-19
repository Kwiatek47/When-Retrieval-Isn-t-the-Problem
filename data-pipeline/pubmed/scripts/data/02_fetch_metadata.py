import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from common import NCBIConfig, clean_text, ncbi_get, read_pmids


def chunks(values: list[str], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def text_at(node, xpath: str) -> str:
    values = node.xpath(xpath)
    if not values:
        return ""
    value = values[0]
    if isinstance(value, etree._Element):
        return clean_text(" ".join(value.itertext()))
    return clean_text(value)


def list_text(node, xpath: str) -> list[str]:
    return [clean_text(" ".join(item.itertext())) for item in node.xpath(xpath)]


def parse_pubmed_xml(xml_bytes: bytes, review_pmids: set[str], systematic_pmids: set[str]) -> list[dict]:
    root = etree.fromstring(xml_bytes)
    rows: list[dict] = []
    for article in root.xpath(".//PubmedArticle"):
        pmid = text_at(article, ".//MedlineCitation/PMID")
        title = text_at(article, ".//Article/ArticleTitle")
        abstract_parts = list_text(article, ".//Article/Abstract/AbstractText")
        abstract = clean_text(" ".join(abstract_parts))
        doi = text_at(article, './/ArticleIdList/ArticleId[@IdType="doi"]')
        journal = text_at(article, ".//Article/Journal/Title")
        year = text_at(article, ".//Article/Journal/JournalIssue/PubDate/Year")
        publication_types = list_text(article, ".//PublicationTypeList/PublicationType")

        rows.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "journal": journal,
                "year": int(year) if year.isdigit() else None,
                "publication_types": publication_types,
                "is_review": pmid in review_pmids,
                "is_systematic": pmid in systematic_pmids,
            }
        )
    return rows


def fetch_xml(pmids: list[str], config: NCBIConfig) -> bytes:
    response = ncbi_get(
        "efetch.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
        config,
    )
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmids", required=True)
    parser.add_argument("--review-pmids", required=True)
    parser.add_argument("--systematic-pmids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    pmids = read_pmids(args.pmids)
    review_pmids = set(read_pmids(args.review_pmids))
    systematic_pmids = set(read_pmids(args.systematic_pmids))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    config = NCBIConfig()
    try:
        for batch_number, batch in enumerate(chunks(pmids, args.batch_size), start=1):
            xml_bytes = fetch_xml(batch, config)
            rows = parse_pubmed_xml(xml_bytes, review_pmids, systematic_pmids)
            print(f"Fetched metadata batch {batch_number}: {len(rows)} records", flush=True)
            if not rows:
                continue
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()

