import argparse
import datetime as dt
import json
from pathlib import Path

from common import NCBIConfig, json_loads_lenient, ncbi_get, write_lines


def date_query(article_type: str, start: dt.date, end: dt.date) -> str:
    return (
        f"({article_type} AND hasabstract AND english[la] "
        f'AND ("{start:%Y/%m/%d}"[dp] : "{end:%Y/%m/%d}"[dp]) '
        "NOT (retracted publication[pt] OR retraction notice[pt]))"
    )


def esearch_count(term: str, config: NCBIConfig) -> int:
    response = ncbi_get(
        "esearch.fcgi",
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": 0},
        config,
    )
    data = json_loads_lenient(response.content)
    return int(data["esearchresult"]["count"])


def esearch_ids(term: str, count: int, config: NCBIConfig, batch_size: int) -> list[str]:
    ids: list[str] = []
    for start in range(0, count, batch_size):
        response = ncbi_get(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "retstart": start,
                "retmax": min(batch_size, count - start),
            },
            config,
        )
        data = json_loads_lenient(response.content)
        ids.extend(data["esearchresult"].get("idlist", []))
    return ids


def fetch_by_uid_range(
    base_term: str,
    label: str,
    start: dt.date,
    end: dt.date,
    uid_min: int,
    uid_max: int,
    config: NCBIConfig,
    max_window_count: int,
    batch_size: int,
) -> list[str]:
    term = f"({base_term} AND {uid_min}:{uid_max} [UID])"
    count = esearch_count(term, config)
    if count == 0:
        print(f"{label}: {start}..{end} uid={uid_min}:{uid_max} -> 0 PMIDs")
        return []
    if count <= max_window_count:
        ids = esearch_ids(term, count, config, batch_size)
        print(f"{label}: {start}..{end} uid={uid_min}:{uid_max} -> fetched {len(ids)} / expected {count} PMIDs")
        return ids
    if uid_min == uid_max:
        raise RuntimeError(f"UID window still too large: {uid_min}:{uid_max} count={count}")
    mid = (uid_min + uid_max) // 2
    return (
        fetch_by_uid_range(base_term, label, start, end, uid_min, mid, config, max_window_count, batch_size)
        + fetch_by_uid_range(base_term, label, start, end, mid + 1, uid_max, config, max_window_count, batch_size)
    )


def fetch_partitioned(
    article_type: str,
    label: str,
    start: dt.date,
    end: dt.date,
    config: NCBIConfig,
    max_window_count: int,
    max_uid: int,
    batch_size: int,
) -> list[str]:
    term = date_query(article_type, start, end)
    count = esearch_count(term, config)
    if count == 0:
        return []
    if count <= max_window_count:
        ids = esearch_ids(term, count, config, batch_size)
        print(f"{label}: {start}..{end} -> fetched {len(ids)} / expected {count} PMIDs")
        return ids
    if start == end:
        return fetch_by_uid_range(term, label, start, end, 1, max_uid, config, max_window_count, batch_size)

    mid = start + (end - start) // 2
    return (
        fetch_partitioned(article_type, label, start, mid, config, max_window_count, max_uid, batch_size)
        + fetch_partitioned(article_type, label, mid + dt.timedelta(days=1), end, config, max_window_count, max_uid, batch_size)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--history-out-dir")
    parser.add_argument("--max-window-count", type=int, default=9999)
    parser.add_argument("--max-uid", type=int, default=60000000)
    parser.add_argument("--batch-size", type=int, default=9999)
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.date_from)
    end = dt.date.fromisoformat(args.date_to)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = NCBIConfig()

    review_pmids = sorted(set(fetch_partitioned("review[pt]", "review", start, end, config, args.max_window_count, args.max_uid, args.batch_size)), key=int)
    systematic_pmids = sorted(set(fetch_partitioned("systematic[sb]", "systematic", start, end, config, args.max_window_count, args.max_uid, args.batch_size)), key=int)
    final_pmids = sorted(set(review_pmids) | set(systematic_pmids), key=int)

    write_lines(out_dir / "review_pmids.txt", review_pmids)
    write_lines(out_dir / "systematic_pmids.txt", systematic_pmids)
    write_lines(args.out, final_pmids)

    summary = {
        "review_pmids": len(review_pmids),
        "systematic_pmids": len(systematic_pmids),
        "final_pmids": len(final_pmids),
    }
    (out_dir / "pmid_fetch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

