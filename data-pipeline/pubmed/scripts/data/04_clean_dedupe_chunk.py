import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-documents", required=True)
    parser.add_argument("--out-chunks", required=True)
    parser.add_argument("--stats-out", required=True)
    parser.add_argument("--duckdb-memory-limit", default="5GB")
    parser.add_argument("--duckdb-temp-dir", default="data/tmp/duckdb")
    args = parser.parse_args()

    Path(args.out_documents).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_chunks).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.duckdb_temp_dir).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{args.duckdb_temp_dir}'")

    print("Reading and cleaning metadata with DuckDB...", flush=True)
    input_rows = con.execute(f"SELECT count(*) FROM read_parquet('{args.input}')").fetchone()[0]

    con.execute(
        f"""
        CREATE TEMP TABLE cleaned AS
        SELECT
          cast(pmid AS VARCHAR) AS pmid,
          nullif(trim(title), '') AS title,
          nullif(trim(abstract), '') AS abstract,
          nullif(lower(trim(doi)), '') AS doi,
          nullif(trim(journal), '') AS journal,
          try_cast(year AS INTEGER) AS year,
          publication_types,
          coalesce(is_review, false) AS is_review,
          coalesce(is_systematic, false) AS is_systematic,
          md5(lower(trim(title))) AS title_hash,
          md5(lower(trim(title || ' ' || abstract))) AS content_hash
        FROM read_parquet('{args.input}')
        WHERE
          pmid IS NOT NULL
          AND length(trim(title)) > 0
          AND length(trim(abstract)) > 0
          AND array_length(regexp_split_to_array(trim(abstract), '\\s+')) >= 50
          AND lower(coalesce(title, '') || ' ' || coalesce(abstract, '')) NOT LIKE '%retracted%'
        """
    )

    cleaned_rows = con.execute("SELECT count(*) FROM cleaned").fetchone()[0]
    print(f"Cleaned rows: {cleaned_rows} / input rows: {input_rows}", flush=True)

    print("Deduplicating by PMID, DOI, title hash, and content hash...", flush=True)
    con.execute(
        """
        CREATE TEMP TABLE dedupe_pmid AS
        SELECT * EXCLUDE rn
        FROM (
          SELECT *, row_number() OVER (PARTITION BY pmid ORDER BY year DESC NULLS LAST) AS rn
          FROM cleaned
        )
        WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE dedupe_doi AS
        SELECT * EXCLUDE rn
        FROM (
          SELECT *, row_number() OVER (PARTITION BY coalesce(doi, pmid) ORDER BY year DESC NULLS LAST) AS rn
          FROM dedupe_pmid
        )
        WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE documents AS
        SELECT * EXCLUDE rn
        FROM (
          SELECT *, row_number() OVER (PARTITION BY title_hash ORDER BY year DESC NULLS LAST) AS rn
          FROM dedupe_doi
        )
        WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE chunks AS
        SELECT
          'pubmed:' || pmid || ':abstract' AS chunk_id,
          pmid,
          title,
          abstract,
          'Title: ' || title || '\n\nAbstract: ' || abstract AS text,
          doi,
          journal,
          year,
          publication_types,
          is_review,
          is_systematic
        FROM documents
        """
    )

    dedupe_pmid_rows = con.execute("SELECT count(*) FROM dedupe_pmid").fetchone()[0]
    dedupe_doi_rows = con.execute("SELECT count(*) FROM dedupe_doi").fetchone()[0]
    final_documents = con.execute("SELECT count(*) FROM documents").fetchone()[0]
    final_chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]

    print(f"Writing documents to {args.out_documents}...", flush=True)
    con.execute(f"COPY documents TO '{args.out_documents}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"Writing chunks to {args.out_chunks}...", flush=True)
    con.execute(f"COPY chunks TO '{args.out_chunks}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    stats = {
        "input_rows": input_rows,
        "cleaned_rows": cleaned_rows,
        "dedupe_pmid_rows": dedupe_pmid_rows,
        "dedupe_doi_rows": dedupe_doi_rows,
        "final_documents": final_documents,
        "final_chunks": final_chunks,
        "reject_stats": {
            "removed_by_cleaning_filters": input_rows - cleaned_rows
        },
        "duplicate_stats": {
            "duplicates_by_pmid": cleaned_rows - dedupe_pmid_rows,
            "duplicates_by_doi": dedupe_pmid_rows - dedupe_doi_rows,
            "duplicates_after_doi": dedupe_doi_rows - final_documents
        },
    }
    Path(args.stats_out).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()

